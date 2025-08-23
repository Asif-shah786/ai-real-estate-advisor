"""
Clean RAG Pipeline for AI Real Estate Advisor - FIXED VERSION

This module provides a clean interface for the RAG pipeline that can be used
by both the Streamlit UI and evaluation code.

Expected interface:
pipeline.run_query(query) => {answer: str, contexts: List[str], meta: dict}
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
    print("⚠️ python-dotenv not installed. Using system environment variables.")

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
from retrieval import CrossEncoderRerankRetriever
from common.cfg import get_openai_api_key, get_config
import utils

# Get configuration
config = get_config()

# Configuration constants from config
config = get_config()
MEMORY_TOKEN_LIMIT = config.memory_token_limit


class RAGPipeline:
    """
    Clean RAG pipeline that can be used independently of the UI.

    This class encapsulates all the RAG logic and provides a clean interface
    for both the Streamlit UI and evaluation code.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline.

        Args:
            openai_api_key: OpenAI API key (optional, will use config if not provided)
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
        self.vectordb = None
        self.qa_chain = None
        self.memory = None
        self.retriever = None
        self.base_retriever = None

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Set up all the RAG components."""
        try:
            # Configure LLM and embedding model directly
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                streaming=True,
                api_key=self.openai_api_key,
                verbose=False,
            )

            self.embedding_model = OpenAIEmbeddings(
                model=config.embedding_model, api_key=self.openai_api_key
            )

            # Setup vector database
            self.vectordb = self._setup_vectordb()

            # Setup retriever
            self.retriever = self._setup_retriever()

            # Setup memory
            self.memory = self._setup_memory()

            # Setup QA chain
            self.qa_chain = self._setup_qa_chain()

            print("✅ RAG Pipeline components initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize RAG Pipeline: {e}")
            traceback.print_exc()
            raise

    def _setup_vectordb(self):
        """Set up the vector database."""
        try:
            from aspect_based_chunker import create_aspect_based_vectordb

            if self.embedding_model is None:
                raise Exception("Embedding model not initialized")

            # Get absolute paths
            project_root = os.path.dirname(os.path.abspath(__file__))
            properties_file = os.path.join(project_root, config.properties_file)
            legal_file = os.path.join(project_root, config.legal_file)

            print(f"🔄 Setting up vector database: {config.db_name}")
            print(f"📁 Using absolute paths: {properties_file}, {legal_file}")

            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(project_root)
            print(f"📁 Changed working directory to: {os.getcwd()}")

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
                print(f"📁 Restored working directory to: {os.getcwd()}")

            if vectordb is None:
                raise Exception("Failed to create vector database")

            print("✅ Vector database setup completed")
            return vectordb

        except Exception as e:
            print(f"❌ Vector database setup failed: {e}")
            raise

    def _setup_retriever(self):
        """Set up the retriever with improved fallback logic."""
        try:
            if self.vectordb is None:
                raise Exception("Vector database not initialized")
            if self.llm is None:
                raise Exception("LLM not initialized")

            # Create basic retriever first (always works)
            basic_retriever = self.vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": config.retrieval_top_n}
            )
            self.base_retriever = basic_retriever
            print("✅ Created basic similarity retriever")

            # Try to enhance with reranking
            try:
                rerank_retriever = CrossEncoderRerankRetriever.from_vectorstore(
                    vectordb=self.vectordb,
                    top_k=config.retrieval_top_k,
                    top_n=config.retrieval_top_n,
                    model_name="BAAI/bge-reranker-large",
                )
                print("✅ Successfully created CrossEncoderRerankRetriever")
                primary_retriever = rerank_retriever
            except Exception as e:
                print(f"⚠️ CrossEncoderRerankRetriever failed: {e}")
                primary_retriever = basic_retriever

            # CRITICAL FIX: Use basic retriever instead of SelfQueryRetriever
            # SelfQueryRetriever often fails to parse queries properly
            print(f"🔍 Using retriever: {type(primary_retriever).__name__}")
            return primary_retriever

        except Exception as e:
            print(f"❌ Retriever setup failed: {e}")
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
            print("✅ Memory setup completed")
            return memory

        except Exception as e:
            print(f"❌ Memory setup failed: {e}")
            raise

    def _setup_qa_chain(self):
        """Set up the QA chain with proper prompt handling."""
        try:
            if self.llm is None or self.retriever is None or self.memory is None:
                raise Exception("Required components not initialized")

            # CRITICAL FIX: Use proper prompt template
            # The key issue is likely in the prompt formatting
            try:
                # Get a working prompt template
                prompt_text = get_prompt_template("standard")

                # Ensure the prompt has the right variables
                if "{context}" not in prompt_text or "{question}" not in prompt_text:
                    # Fallback to a known working prompt
                    prompt_text = """You are a helpful AI assistant for real estate queries. 
Use the following context to answer the question. If you cannot answer based on the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""

                prompt_template = PromptTemplate.from_template(prompt_text)
                print("✅ Created prompt template successfully")

            except Exception as e:
                print(f"⚠️ Error creating prompt template: {e}")
                # Use a simple fallback prompt
                prompt_template = PromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                )

            # CRITICAL FIX: Use ConversationalRetrievalChain properly
            try:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,  # Enable verbose for debugging
                    combine_docs_chain_kwargs={"prompt": prompt_template},
                    # CRITICAL: Set proper chain type
                    chain_type="stuff",  # Use 'stuff' chain type for better control
                )
                print("✅ Successfully created ConversationalRetrievalChain")
                return qa_chain

            except Exception as e:
                print(f"❌ Failed to create ConversationalRetrievalChain: {e}")
                raise

        except Exception as e:
            print(f"❌ QA chain setup failed: {e}")
            raise

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

            print(f"🔍 Processing query: {query[:100]}...")

            # Clear memory if not using it
            if not use_memory and self.memory:
                self.memory.clear()

            # CRITICAL FIX: Test retrieval first
            try:
                # Test if retriever works
                if self.retriever:
                    test_docs = self.retriever.get_relevant_documents(query)
                    print(f"🔍 Retriever returned {len(test_docs)} documents")

                    if len(test_docs) == 0:
                        print("⚠️ No documents retrieved, trying base retriever")
                        if self.base_retriever:
                            test_docs = self.base_retriever.get_relevant_documents(
                                query
                            )
                            print(
                                f"🔍 Base retriever returned {len(test_docs)} documents"
                            )

            except Exception as retrieval_error:
                print(f"⚠️ Retrieval test failed: {retrieval_error}")
                # Continue anyway, the chain might handle it

            # Process the query
            if callbacks:
                result = self.qa_chain.invoke(
                    {"question": query}, {"callbacks": callbacks}
                )
            else:
                result = self.qa_chain.invoke({"question": query})

            print(f"✅ QA chain completed")

            # Extract and validate results
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])

            # CRITICAL FIX: Ensure we have contexts
            if not source_docs and self.base_retriever:
                print("🔄 No source documents, using base retriever as fallback")
                try:
                    fallback_docs = self.base_retriever.get_relevant_documents(query)
                    source_docs = fallback_docs
                    print(f"🔄 Fallback retrieved {len(source_docs)} documents")
                except Exception as fallback_error:
                    print(f"⚠️ Fallback retrieval failed: {fallback_error}")

            contexts = [doc.page_content for doc in source_docs]

            print(f"📊 Result summary:")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Contexts: {len(contexts)} documents")
            print(f"   Answer preview: {answer[:100]}...")

            # Prepare metadata
            meta = {
                "source_count": len(source_docs),
                "retriever_type": str(type(self.retriever)),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "source_metadata": (
                    [doc.metadata for doc in source_docs] if source_docs else []
                ),
            }

            return {
                "answer": answer,
                "contexts": contexts,
                "meta": meta,
            }

        except Exception as e:
            print(f"❌ Error in run_query: {e}")
            traceback.print_exc()

            # Try a simple fallback approach
            try:
                print("🔄 Attempting simple fallback...")
                if self.base_retriever and self.llm:
                    # Get documents directly
                    docs = self.base_retriever.get_relevant_documents(query)
                    contexts = [doc.page_content for doc in docs]

                    # Create a simple prompt
                    context_text = "\n\n".join(contexts[:3])  # Use top 3 contexts
                    simple_prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""

                    # Get answer directly from LLM
                    response = self.llm.invoke(simple_prompt)
                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    print(
                        f"✅ Fallback successful: {len(answer)} char answer, {len(contexts)} contexts"
                    )

                    return {
                        "answer": answer,
                        "contexts": contexts,
                        "meta": {
                            "fallback": True,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "query": query,
                            "source_count": len(docs),
                        },
                    }

            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")

            # Final error response
            return {
                "answer": f"Error processing query: {str(e)}",
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
            print("✅ Memory cleared")

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
        }

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get detailed information about the retriever configuration."""
        if self.retriever is None:
            return {"status": "Not initialized"}

        retriever_type = str(type(self.retriever))
        info: Dict[str, Any] = {
            "type": retriever_type,
            "status": "Active",
        }

        if "CrossEncoderRerankRetriever" in retriever_type:
            info["model"] = "BAAI/bge-reranker-large"
            info["top_k"] = config.retrieval_top_k
            info["top_n"] = config.retrieval_top_n
            info["description"] = "Advanced reranking with cross-encoder model"
        else:
            info["description"] = "Basic similarity search"
            info["search_kwargs"] = {"k": config.retrieval_top_n}

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
