"""
Clean RAG Pipeline for AI Real Estate Advisor - FIXED VERSION

This module provides a clean interface for the RAG pipeline that can be used
by both the Streamlit UI and evaluation code.

Expected interface:
pipeline.run_query(query) => {answer: str, contexts: List[str], meta: dict}

IMPORTANT: If you encounter NumPy compatibility errors with PyTorch/sentence-transformers,
run: pip install "numpy<2.0" to downgrade NumPy to a compatible version.
"""

import os
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(" python-dotenv not installed. Using system environment variables.")

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.schema.retriever import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from prompts import CONTEXTUALIZATION_SYSTEM_PROMPT, get_prompt_template
from common.cfg import get_openai_api_key, get_config
import utils

# Get configuration
config = get_config()
MEMORY_TOKEN_LIMIT = config.memory_token_limit


class RAGPipeline:
    """
    Clean RAG pipeline that can be used independently of the UI.

    This class encapsulates all the RAG logic and provides a clean interface
    for both the Streamlit UI and evaluation code.
    """

    def __init__(self, openai_api_key: Optional[str] = None, existing_vectordb=None):
        """
        Initialize the RAG pipeline.

        Args:
            openai_api_key: OpenAI API key (optional, will use config if not provided)
            existing_vectordb: Existing vector database to use (optional, will create new if not provided)
        """
        # Use provided API key or get from config
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = get_openai_api_key()

        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as parameter."
            )

        self.llm = None
        self.embedding_model = None
        self.vectordb = existing_vectordb  # Use existing database if provided
        self.qa_chain = None
        self.memory = None
        self.retriever = None
        self.base_retriever = None
        self.self_query_retriever = None

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Set up all the RAG components."""
        try:
            # Configure LLM and embedding model directly
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from pydantic import SecretStr

            # Get API key from config
            api_key = get_openai_api_key()
            secret_api_key = SecretStr(api_key)

            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                streaming=True,
                api_key=secret_api_key,
                verbose=False,
            )

            self.embedding_model = OpenAIEmbeddings(
                model=config.embedding_model, api_key=secret_api_key
            )

            # Setup vector database only if we don't have one
            if self.vectordb is None:
                self.vectordb = self._setup_vectordb()

            # Setup retriever
            self.retriever = self._setup_retriever()

            # Setup memory
            self.memory = self._setup_memory()

            # Setup QA chain
            self.qa_chain = self._setup_qa_chain()

            print("RAG Pipeline components initialized successfully")

        except Exception as e:
            print(f"Failed to initialize RAG Pipeline: {e}")
            traceback.print_exc()
            raise

    def _setup_vectordb(self):
        """Set up the vector database."""
        try:
            # If we already have a database, use it
            if self.vectordb is not None:
                print(f"Using existing vector database (no recreation needed)")
                return self.vectordb

            from aspect_based_chunker import create_aspect_based_vectordb

            if self.embedding_model is None:
                raise Exception("Embedding model not initialized")

            # Get absolute paths
            project_root = os.path.dirname(os.path.abspath(__file__))
            properties_file = os.path.join(project_root, config.properties_file)
            legal_file = os.path.join(project_root, config.legal_file)

            print(f" Setting up vector database: {config.db_name}")
            print(f"Using absolute paths: {properties_file}, {legal_file}")

            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(project_root)
            print(f"Changed working directory to: {os.getcwd()}")

            try:
                vectordb = create_aspect_based_vectordb(
                    openai_api_key=self.openai_api_key,
                    properties_file=properties_file,
                    legal_file=legal_file,
                    embedding_model=self.embedding_model,
                    force_recreate=False,
                )
            finally:
                os.chdir(original_cwd)
                print(f"Restored working directory to: {os.getcwd()}")

            if vectordb is None:
                raise Exception("Failed to create vector database")

            print("Vector database setup completed")
            return vectordb

        except Exception as e:
            print(f"Vector database setup failed: {e}")
            raise

    def _setup_retriever(self):
        """Set up the retriever with FIXED search strategy."""
        try:
            if self.vectordb is None:
                raise Exception("Vector database not initialized")
            if self.llm is None:
                raise Exception("LLM not initialized")

            # CRITICAL FIX: Use MMR (Maximum Marginal Relevance) search
            # This provides better diversity and relevance than pure similarity
            mmr_retriever = self.vectordb.as_retriever(
                search_type="mmr",  # Changed from "similarity" to "mmr"
                search_kwargs={
                    "k": config.retrieval_top_n,
                    "fetch_k": config.retrieval_top_n * 3,  # Fetch 3x for MMR diversity
                    "lambda_mult": 0.7,  # Balance between relevance and diversity
                },
            )
            self.base_retriever = mmr_retriever
            print("Created MMR retriever for better relevance")

            # CRITICAL FIX: Add SelfQueryRetriever for structured queries
            # This is essential for handling price, bedroom, postcode filters
            try:
                from langchain.chains.query_constructor.schema import AttributeInfo
                from langchain.retrievers.self_query.base import SelfQueryRetriever

                metadata_field_info = [
                    AttributeInfo(
                        name="price_int",
                        description="Property price in British pounds (exact number)",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="bedrooms",
                        description="Number of bedrooms (exact number)",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="property_type",
                        description="Property type: detached, semi_detached, terraced, flat, bungalow, etc.",
                        type="string",
                    ),
                    AttributeInfo(
                        name="postcode",
                        description="Postcode area like M4, WN3, etc.",
                        type="string",
                    ),
                    AttributeInfo(
                        name="tenure",
                        description="Property tenure: freehold or leasehold",
                        type="string",
                    ),
                    AttributeInfo(
                        name="type",
                        description="Content type: overview, crime, schools, transport",
                        type="string",
                    ),
                ]

                document_content_description = "Real estate property listings with price, bedrooms, property type, and location details"

                # Create SelfQueryRetriever with correct parameters
                self_query_retriever = SelfQueryRetriever.from_llm(
                    self.llm,
                    self.vectordb,
                    document_content_description,
                    metadata_field_info,
                    enable_limit=True,
                    search_kwargs={"k": config.retrieval_top_n},
                    verbose=True,  # Enable to debug query parsing
                    # CRITICAL: Add structured query examples for better parsing
                    examples=[
                        {
                            "query": "Show me properties under £500,000",
                            "filter": "price_int < 500000",
                        },
                        {
                            "query": "3-bedroom houses in M4",
                            "filter": "bedrooms == 3.0 AND postcode == 'M4'",
                        },
                        {
                            "query": "Detached properties in Manchester",
                            "filter": "property_type == 'detached'",
                        },
                    ],
                )
                self.self_query_retriever = self_query_retriever
                print("Created SelfQueryRetriever for structured queries")

                # Use SelfQueryRetriever as primary
                primary_retriever = self_query_retriever
                print("Using SelfQueryRetriever as primary retriever")

            except Exception as e:
                print(f" SelfQueryRetriever failed: {e}")
                print(" Using MMR retriever as fallback")
                primary_retriever = mmr_retriever

            return primary_retriever

        except Exception as e:
            print(f"Retriever setup failed: {e}")
            raise

    def _setup_memory(self):
        """Set up conversation memory."""
        try:
            if self.llm is None:
                raise Exception("LLM not initialized")

            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                max_token_limit=MEMORY_TOKEN_LIMIT,
            )
            print("Memory setup completed")
            return memory

        except Exception as e:
            print(f"Memory setup failed: {e}")
            raise

    def _setup_qa_chain(self):
        """Set up the QA chain with RAGAS-optimized prompt."""
        try:
            if self.llm is None or self.retriever is None or self.memory is None:
                raise Exception("Required components not initialized")

            # CRITICAL FIX: Use RAGAS-optimized prompt template
            try:
                # Create a prompt that works well with RAGAS evaluation
                ragas_optimized_prompt = """You are a helpful real estate assistant. Use the provided context to answer the question accurately and completely.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the requested information, clearly state this
3. For property listings, include: price, bedrooms, location, property type
4. Be specific and factual in your responses
5. If no relevant properties are found, suggest alternative searches
6. Format your response clearly with bullet points or structured lists when showing multiple properties

Context:
{context}

Question: {question}

Answer:"""

                prompt_template = PromptTemplate.from_template(ragas_optimized_prompt)
                print("Created RAGAS-optimized prompt template")

            except Exception as e:
                print(f" Error creating prompt template: {e}")
                # Fallback to simple prompt
                prompt_template = PromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                )

            # Create ConversationalRetrievalChain
            try:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    combine_docs_chain_kwargs={"prompt": prompt_template},
                    chain_type="stuff",
                )
                print(
                    "Successfully created ConversationalRetrievalChain with RAGAS optimization"
                )
                print(f"Using prompt template: {type(prompt_template).__name__}")
                print(f"Prompt variables: {prompt_template.input_variables}")
                return qa_chain

            except Exception as e:
                print(f"Failed to create ConversationalRetrievalChain: {e}")
                raise

        except Exception as e:
            print(f"QA chain setup failed: {e}")
            raise

    def _get_relevant_documents_with_fallback(self, query: str) -> List:
        """Get relevant documents with smart fallback strategy."""
        docs = []

        # CRITICAL: Detect if this is a structured query that should use SelfQueryRetriever
        structured_query_indicators = [
            "under £",
            "over £",
            "between £",
            "£",  # Price filters
            "bedroom",
            "bed",
            "beds",  # Bedroom filters
            "in M",
            "postcode",
            "area",  # Location filters
            "detached",
            "semi",
            "terraced",
            "flat",
            "bungalow",  # Property type
            "freehold",
            "leasehold",  # Tenure filters
        ]

        is_structured_query = any(
            indicator in query.lower() for indicator in structured_query_indicators
        )

        if is_structured_query and self.self_query_retriever:
            print(f"Detected structured query: '{query[:50]}...'")
            print(f"Using SelfQueryRetriever for metadata-based filtering")

            try:
                docs = self.self_query_retriever.invoke(query)
                if docs:
                    print(f"SelfQueryRetriever found {len(docs)} documents")
                    print(
                        f"SelfQueryRetriever query type: 'structured' (good for filters)"
                    )

                    # Log metadata filtering results
                    if docs and hasattr(docs[0], "metadata"):
                        print(f"Sample metadata from SelfQueryRetriever:")
                        sample_meta = docs[0].metadata
                        for key in [
                            "price_int",
                            "bedrooms",
                            "postcode",
                            "property_type",
                        ]:
                            if key in sample_meta:
                                print(f"   {key}: {sample_meta[key]}")

                    return docs
                else:
                    print(" SelfQueryRetriever returned no documents")
                    print(
                        f"SelfQueryRetriever query type: 'structured' (may be too specific)"
                    )
            except Exception as e:
                print(f" SelfQueryRetriever failed: {e}")
                print(f"SelfQueryRetriever error type: {type(e).__name__}")
                print(f" Falling back to semantic search...")

        # Strategy 2: Try primary retriever (usually SelfQueryRetriever)
        if self.retriever and self.retriever != self.self_query_retriever:
            try:
                docs = self.retriever.invoke(query)
                if docs:
                    print(f"Primary retriever found {len(docs)} documents")
                    print(f"Primary retriever type: {type(self.retriever).__name__}")
                    return docs
            except Exception as e:
                print(f" Primary retriever failed: {e}")

        # Strategy 3: Fall back to MMR base retriever (semantic similarity)
        if self.base_retriever:
            try:
                docs = self.base_retriever.invoke(query)
                print(f"MMR base retriever found {len(docs)} documents")
                print(
                    f"MMR retriever: 'semantic similarity' (good for general queries)"
                )
                return docs
            except Exception as e:
                print(f" MMR base retriever failed: {e}")

        return docs

    def run_query(
        self, query: str, use_memory: bool = True, callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run a query through the RAG pipeline.

        Args:
            query: The user's question
            use_memory: Whether to use conversation memory (default: True)
            callbacks: Optional list of callbacks for streaming

        Returns:
            Dict with answer, contexts, and metadata
        """
        try:
            if not self.qa_chain:
                raise Exception("QA chain not initialized")

            print(f"Processing query: {query[:100]}...")

            # Clear memory if not using it
            if not use_memory and self.memory:
                self.memory.clear()

            # CRITICAL FIX: Enhanced retrieval testing and fallback
            try:
                test_docs = self._get_relevant_documents_with_fallback(query)
                contexts_for_chain = [
                    doc.page_content for doc in test_docs if doc.page_content.strip()
                ]

                # Log retrieval results for debugging
                print(
                    f"Retrieval results: {len(test_docs)} docs, {len(contexts_for_chain)} valid contexts"
                )
                for i, doc in enumerate(test_docs[:3]):  # Show first 3 docs
                    print(
                        f"   Retrieved Doc {i+1}: content_length={len(doc.page_content)}, has_content={bool(doc.page_content.strip())}"
                    )

            except Exception as retrieval_error:
                print(f" Enhanced retrieval failed: {retrieval_error}")
                # Fallback to basic retrieval
                try:
                    if self.base_retriever:
                        test_docs = self.base_retriever.invoke(query)
                        contexts_for_chain = [
                            doc.page_content
                            for doc in test_docs
                            if doc.page_content.strip()
                        ]
                        print(
                            f" Basic retrieval fallback: {len(test_docs)} docs, {len(contexts_for_chain)} valid contexts"
                        )
                    else:
                        contexts_for_chain = []
                        test_docs = []
                except Exception as basic_error:
                    print(f" Basic retrieval also failed: {basic_error}")
                    contexts_for_chain = []
                    test_docs = []

            print(f"Pre-retrieval test: {len(contexts_for_chain)} valid contexts")

            # Process the query through the chain
            try:
                # CRITICAL DEBUG: Test what the chain's retriever is actually doing
                if self.retriever and hasattr(self.retriever, "invoke"):
                    print(f"Testing chain's retriever: {type(self.retriever).__name__}")
                    try:
                        chain_test_docs = self.retriever.invoke(query)
                        print(f"Chain retriever test: {len(chain_test_docs)} docs")
                        for i, doc in enumerate(chain_test_docs[:2]):
                            print(
                                f"   Chain Doc {i+1}: content_length={len(doc.page_content)}, has_content={bool(doc.page_content.strip())}"
                            )
                    except Exception as retriever_test_error:
                        print(f" Chain retriever test failed: {retriever_test_error}")
                else:
                    print(" Chain retriever is None or missing invoke method!")

                if callbacks:
                    result = self.qa_chain.invoke(
                        {"question": query}, {"callbacks": callbacks}
                    )
                else:
                    result = self.qa_chain.invoke({"question": query})

                print("QA chain completed successfully")

                # Extract results
                answer = result.get("answer", "")
                source_docs = result.get("source_documents", [])

                # CRITICAL FIX: Use chain's source_documents, not our pre-retrieved docs
                # The chain's source_documents are the ones that actually get processed
                if source_docs:
                    contexts = [doc.page_content for doc in source_docs]
                    print(f"Chain returned {len(source_docs)} source documents")

                    # Log each source document to debug
                    for i, doc in enumerate(source_docs):
                        print(
                            f"   Doc {i+1}: content_length={len(doc.page_content)}, metadata={doc.metadata}"
                        )
                        if not doc.page_content.strip():
                            print(f"    WARNING: Doc {i+1} has EMPTY page_content!")
                else:
                    print(" Chain returned NO source documents, using fallback")
                    # Fallback to our pre-retrieved contexts
                    contexts = contexts_for_chain
                    source_docs = test_docs

            except Exception as chain_error:
                print(f" QA chain failed: {chain_error}")
                # Fall back to direct LLM call with pre-retrieved contexts
                if contexts_for_chain and self.llm:
                    print(" Using direct LLM fallback...")
                    context_text = "\n\n".join(contexts_for_chain[:5])

                    direct_prompt = f"""Based on the following real estate context, provide a helpful and accurate answer to the question.

Context:
{context_text}

Question: {query}

Instructions:
- Use only the information provided in the context
- Be specific about property details (price, bedrooms, location, type)
- If the context doesn't contain the requested information, clearly state this
- Provide actionable information where possible

Answer:"""

                    response = self.llm.invoke(direct_prompt)
                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    contexts = contexts_for_chain
                    source_docs = test_docs

                    print(f"Direct LLM fallback successful")
                else:
                    raise chain_error

            print(f"Result summary:")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Contexts: {len(contexts)} documents")
            print(f"   Answer preview: {answer[:100]}...")

            # ENHANCED LOGGING: Show exactly what's being passed to evaluation
            print(f"DETAILED CONTEXT ANALYSIS:")
            print(f"   Total contexts retrieved: {len(contexts)}")
            print(
                f"   Contexts source: {'chain source_documents' if source_docs and len(source_docs) == len(contexts) else 'fallback test_docs'}"
            )

            for i, context in enumerate(contexts[:3]):  # Show first 3 contexts
                print(f"   Context {i+1} length: {len(context)} chars")
                print(f"   Context {i+1} preview: {context[:150]}...")
                print(
                    f"   Context {i+1} contains property info: {'price' in context.lower() or 'bedroom' in context.lower()}"
                )

            # CRITICAL: Check if any contexts are empty
            empty_contexts = [i for i, ctx in enumerate(contexts) if not ctx.strip()]
            if empty_contexts:
                print(f" WARNING: Empty contexts found at positions: {empty_contexts}")
                print(f" This will cause RAGAS evaluation to fail!")

            # Show metadata for debugging
            if source_docs:
                print(f"SOURCE DOCUMENT METADATA:")
                for i, doc in enumerate(source_docs[:3]):
                    print(f"   Doc {i+1} metadata: {doc.metadata}")
                    print(f"   Doc {i+1} content length: {len(doc.page_content)}")

            # Validate answer quality
            print(f"ANSWER QUALITY CHECK:")

            # Ensure answer is a string for analysis
            answer_str = str(answer) if answer else ""

            print(
                f"   Answer contains property details: {'bedroom' in answer_str.lower() or 'price' in answer_str.lower()}"
            )
            print(
                f"   Answer contains location info: {'manchester' in answer_str.lower() or 'm4' in answer_str.lower()}"
            )
            print(
                f"   Answer is specific: {len(answer_str.split()) > 20}"
            )  # More than 20 words

            # Prepare metadata
            meta = {
                "source_count": len(source_docs),
                "retriever_type": str(type(self.retriever)),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "source_metadata": (
                    [doc.metadata for doc in source_docs] if source_docs else []
                ),
                "retrieval_strategy": "enhanced_fallback",
            }

            return {
                "answer": answer,
                "contexts": contexts,
                "meta": meta,
            }

        except Exception as e:
            print(f"Error in run_query: {e}")
            traceback.print_exc()

            # Final emergency fallback
            try:
                print(" Attempting emergency fallback...")
                emergency_docs = self._get_relevant_documents_with_fallback(query)
                if emergency_docs and self.llm:
                    emergency_contexts = [doc.page_content for doc in emergency_docs]
                    context_text = "\n\n".join(emergency_contexts[:3])

                    emergency_prompt = f"""Answer this real estate question based on the context provided:

Context: {context_text}

Question: {query}

Answer:"""

                    response = self.llm.invoke(emergency_prompt)
                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    return {
                        "answer": answer,
                        "contexts": emergency_contexts,
                        "meta": {
                            "fallback": "emergency",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "query": query,
                            "source_count": len(emergency_docs),
                        },
                    }
            except Exception as emergency_error:
                print(f"Emergency fallback failed: {emergency_error}")

            # Absolute final error response
            return {
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "contexts": [],
                "meta": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                },
            }

    def clear_memory(self):
        """Clear the conversation memory."""
        if self.memory:
            self.memory.clear()
            print("Memory cleared")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            "retriever_type": str(type(self.retriever)),
            "memory_type": str(type(self.memory)),
            "qa_chain_type": str(type(self.qa_chain)),
            "vectordb_type": str(type(self.vectordb)),
            "llm_model": str(self.llm) if self.llm else "Not configured",
            "embedding_model": (
                str(self.embedding_model) if self.embedding_model else "Not configured"
            ),
            "memory_token_limit": MEMORY_TOKEN_LIMIT,
            "supports_streaming": True,
            "has_self_query": self.self_query_retriever is not None,
        }

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get detailed information about the retriever configuration."""
        if self.retriever is None:
            return {"status": "Not initialized"}

        retriever_type = str(type(self.retriever))
        info = {
            "type": retriever_type,
            "status": "Active",
            "has_self_query": self.self_query_retriever is not None,
            "has_mmr_base": self.base_retriever is not None,
        }

        if "SelfQueryRetriever" in retriever_type:
            info.update(
                {
                    "description": "Structured query retrieval with metadata filtering",
                    "supports": "price, bedrooms, property_type, postcode, tenure filters",
                }
            )
        elif "CrossEncoderRerankRetriever" in retriever_type:
            info.update(
                {
                    "model": "BAAI/bge-reranker-large",
                    "top_k": config.retrieval_top_k,
                    "top_n": config.retrieval_top_n,
                    "description": "Advanced reranking with cross-encoder model",
                }
            )
        else:
            info.update(
                {
                    "description": "MMR similarity search with diversity",
                    "search_kwargs": {"k": config.retrieval_top_n},
                }
            )

        return info


def create_rag_pipeline(openai_api_key: Optional[str] = None) -> RAGPipeline:
    """
    Create a new RAG pipeline instance.

    Args:
        openai_api_key: OpenAI API key (optional, will use config if not provided)

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(openai_api_key=openai_api_key)
