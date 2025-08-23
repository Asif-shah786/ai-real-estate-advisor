"""
Clean RAG Pipeline for AI Real Estate Advisor

This module provides a clean interface for the RAG pipeline that can be used
by both the Streamlit UI and evaluation code.

Expected interface:
pipeline.run_query(query) => {answer: str, contexts: List[str], meta: dict}
"""

import os
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever

from prompts import CONTEXTUALIZATION_SYSTEM_PROMPT, get_prompt_template
from retrieval import CrossEncoderRerankRetriever
import utils


class RAGPipeline:
    """
    Clean RAG pipeline that can be used independently of the UI.

    This class encapsulates all the RAG logic and provides a clean interface
    for both the Streamlit UI and evaluation code.
    """

    def __init__(self, openai_api_key: str):
        """
        Initialize the RAG pipeline.

        Args:
            openai_api_key: OpenAI API key for LLM and embeddings
        """
        self.openai_api_key = openai_api_key
        self.llm = None
        self.embedding_model = None
        self.vectordb = None
        self.qa_chain = None
        self.memory = None
        self.retriever = None

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Set up all the RAG components."""
        try:
            # Configure LLM and embedding model
            self.llm = utils.configure_llm()
            self.embedding_model = utils.configure_embedding_model()

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

            vectordb = create_aspect_based_vectordb(
                openai_api_key=self.openai_api_key,
                properties_file="datasets/run_ready_904.json",
                legal_file="datasets/legal_uk_greater_manchester.jsonl",
                embedding_model=self.embedding_model,
                force_recreate=False,  # Use existing database if available
            )

            if vectordb is None:
                raise Exception("Failed to create vector database")

            print("✅ Vector database setup completed")
            return vectordb

        except Exception as e:
            print(f"❌ Vector database setup failed: {e}")
            raise

    def _setup_retriever(self):
        """Set up the retriever with fallback logic."""
        try:
            # Create the primary retriever - prefer reranker, fallback to basic
            primary_retriever = None

            # Try to create CrossEncoderRerankRetriever as the primary choice
            try:
                primary_retriever = CrossEncoderRerankRetriever.from_vectorstore(
                    vectordb=self.vectordb,
                    top_k=20,  # Get more candidates for reranking
                    top_n=5,  # Return top 5 after reranking
                    model_name="BAAI/bge-reranker-large",
                )
                print("✅ Successfully created CrossEncoderRerankRetriever (Primary)")

            except Exception as e:
                print(f"⚠️ CrossEncoderRerankRetriever failed: {e}")
                print("🔄 Falling back to basic similarity retriever")
                # Fallback to basic retriever
                primary_retriever = self.vectordb.as_retriever(
                    search_type="similarity", search_kwargs={"k": 5}
                )

            # Now use the primary retriever consistently
            base_retriever = primary_retriever

            # Log which retriever type we're using
            retriever_type = (
                "CrossEncoderRerankRetriever"
                if "CrossEncoderRerankRetriever" in str(type(primary_retriever))
                else "Basic Similarity Retriever"
            )
            print(f"🔍 Using retriever: {retriever_type}")

            # Try to enhance with SelfQueryRetriever, but fall back gracefully
            try:
                from langchain.chains.query_constructor.schema import AttributeInfo
                from langchain.retrievers.self_query.base import SelfQueryRetriever

                metadata_field_info = [
                    AttributeInfo(
                        name="price_int",
                        description="Listing price in British pounds",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="bedrooms",
                        description="Number of bedrooms in the property",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="bathrooms",
                        description="Number of bathrooms in the property",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="property_type",
                        description="Type of property such as detached, semi-detached, terraced or apartment",
                        type="string",
                    ),
                    AttributeInfo(
                        name="postcode",
                        description="Postcode prefix where the property is located",
                        type="string",
                    ),
                    AttributeInfo(
                        name="tenure",
                        description="Property tenure for example freehold or leasehold",
                        type="string",
                    ),
                ]

                document_content_description = (
                    "Property listings with fields such as price_int, bedrooms, bathrooms, "
                    "property_type, postcode and tenure"
                )

                # Try to create SelfQueryRetriever with proper error handling
                retriever = SelfQueryRetriever.from_llm(
                    self.llm,
                    self.vectordb,
                    document_content_description,
                    metadata_field_info,
                    enable_limit=True,
                    search_kwargs={"k": 20},  # Increased for reranking compatibility
                    verbose=False,  # Reduce verbosity to avoid issues
                )
                print("✅ Successfully created SelfQueryRetriever")

            except Exception as e:
                print(f"⚠️ SelfQueryRetriever failed: {e}")
                print("🔄 Using basic similarity retriever instead")
                # Fallback to the basic retriever we created earlier
                retriever = base_retriever

            return retriever

        except Exception as e:
            print(f"❌ Retriever setup failed: {e}")
            raise

    def _setup_memory(self):
        """Set up conversation memory."""
        try:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                max_token_limit=2000,  # Reasonable limit for memory
            )
            print("✅ Memory setup completed")
            return memory

        except Exception as e:
            print(f"❌ Memory setup failed: {e}")
            raise

    def _setup_qa_chain(self):
        """Set up the QA chain."""
        try:
            # Create history-aware retriever with better error handling
            try:
                # Use contextualization prompt from prompts file
                contextualize_q_system_prompt = CONTEXTUALIZATION_SYSTEM_PROMPT
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                # Create the history-aware retriever chain using LCEL
                history_aware_retriever_chain = create_history_aware_retriever(
                    llm=self.llm,
                    retriever=self.retriever,
                    prompt=contextualize_q_prompt,
                )

                print("✅ Successfully created history-aware retriever using LCEL")
                history_aware_retriever = history_aware_retriever_chain

            except Exception as e:
                print(f"⚠️ History-aware retriever failed: {e}")
                print("🔄 Using basic retriever without history awareness")
                history_aware_retriever = self.retriever

            # Setup QA chain with comprehensive error handling
            try:
                # For LCEL-based history-aware retriever, we need to configure the chain differently
                if hasattr(history_aware_retriever, "invoke"):
                    # This is an LCEL chain, configure ConversationalRetrievalChain to work with it
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,  # Use the base retriever directly
                        memory=self.memory,
                        return_source_documents=True,
                        verbose=False,
                        combine_docs_chain_kwargs={
                            "prompt": PromptTemplate.from_template(
                                get_prompt_template("lcel")
                            )
                        },
                    )
                    print(
                        "✅ Successfully created ConversationalRetrievalChain with LCEL integration"
                    )
                    return qa_chain
                else:
                    # Fallback to standard configuration
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=history_aware_retriever,
                        memory=self.memory,
                        return_source_documents=True,
                        verbose=False,
                        combine_docs_chain_kwargs={
                            "prompt": PromptTemplate.from_template(
                                get_prompt_template("standard")
                            )
                        },
                    )
                    print(
                        "✅ Successfully created ConversationalRetrievalChain with standard retriever"
                    )
                    return qa_chain

            except Exception as e:
                print(f"❌ Failed to create ConversationalRetrievalChain: {e}")
                print("🔄 Attempting fallback configuration...")

                # Fallback: Create a simpler chain configuration
                try:
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,
                        memory=self.memory,
                        return_source_documents=True,
                        verbose=False,
                        combine_docs_chain_kwargs={
                            "prompt": PromptTemplate.from_template(
                                get_prompt_template("fallback")
                            )
                        },
                    )
                    print(
                        "✅ Successfully created fallback ConversationalRetrievalChain"
                    )
                    return qa_chain

                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    raise Exception(
                        f"Could not create QA chain. Original error: {e}, Fallback error: {fallback_error}"
                    )

        except Exception as e:
            print(f"❌ QA chain setup failed: {e}")
            raise

    def run_query(self, query: str, use_memory: bool = True) -> Dict[str, Any]:
        """
        Run a query through the RAG pipeline.

        This is the main interface expected by evaluation code.

        Args:
            query: The user's question
            use_memory: Whether to use conversation memory (default: True)

        Returns:
            Dict with the following structure:
            {
                "answer": str,           # The AI's response
                "contexts": List[str],   # List of retrieved document contents
                "meta": Dict             # Additional metadata
            }
        """
        try:
            if not self.qa_chain:
                raise Exception("QA chain not initialized")

            # Clear memory if not using it
            if not use_memory and self.memory:
                self.memory.clear()

            # Process the query
            result = self.qa_chain.invoke({"question": query})

            # Extract source documents
            source_docs = result.get("source_documents", [])
            contexts = [doc.page_content for doc in source_docs]

            # Extract metadata from source documents
            meta = {
                "source_count": len(source_docs),
                "retriever_type": str(type(self.retriever)),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "source_metadata": (
                    [doc.metadata for doc in source_docs] if source_docs else []
                ),
            }

            # Return in the expected format
            return {
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "meta": meta,
            }

        except Exception as e:
            print(f"❌ Error in run_query: {e}")
            traceback.print_exc()

            # Return error response in expected format
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
        }


# Factory function for easy creation
def create_rag_pipeline(openai_api_key: str) -> RAGPipeline:
    """
    Create a new RAG pipeline instance.

    Args:
        openai_api_key: OpenAI API key

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(openai_api_key)
