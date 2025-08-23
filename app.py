"""
AI Real Estate Advisor - RAG Implementation

This file implements the AI Real Estate Advisor, which uses
a Retrieval-Augmented Generation (RAG) approach with vector databases and
ConversationalRetrievalChain from LangChain.

This version:
1. Uses vector databases for efficient information retrieval
2. Supports multiple LLM models (OpenAI GPT, Llama)
3. Implements streaming responses for better user experience
4. Provides source references for transparency
"""

import os  # For path operations and environment variables
import utils  # Custom utilities for the application
import traceback  # For detailed error tracking
import streamlit as st  # Web UI framework
from streaming import StreamHandler  # Custom handler for streaming responses
from common.cfg import *  # Import configuration variables
from prompts import (
    QUERY_REWRITING_PROMPT,
    CONTEXTUALIZATION_SYSTEM_PROMPT,
    get_prompt_template,
)  # Import prompts from dedicated file
from rag_pipeline import create_rag_pipeline  # Import clean RAG pipeline
from langchain.memory import (
    ConversationSummaryBufferMemory,
)  # For storing conversation context
from langchain.chains import ConversationalRetrievalChain  # Main RAG chain
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)  # For rewriting follow-up questions
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # Prompt utilities for conversational retriever
from langchain.prompts import PromptTemplate  # For identity question prompt
from langchain.schema.retriever import BaseRetriever  # Base class for custom retriever
import pandas as pd  # For data manipulation
from langchain_core.documents.base import Document  # LangChain document structure
from langchain_community.document_loaders.csv_loader import (
    CSVLoader,
)  # For loading local CSV files
from langchain_community.document_loaders.dataframe import (
    DataFrameLoader,
)  # For loading pandas DataFrames
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # For chunking documents
from langchain_community.vectorstores import (
    DocArrayInMemorySearch,
)  # In-memory vector store
import json  # For saving chat sessions
import time  # For timestamps
from datetime import datetime  # For formatted timestamps
from typing import Optional  # For optional parameters
from langchain_openai import ChatOpenAI  # Small LLM for query rewriting
from pydantic import SecretStr  # For handling sensitive API keys

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate

# Configure the Streamlit page
st.set_page_config(
    page_title="🦾 AI Real Estate Advisor", page_icon="💬", layout="wide"
)
# Header will be displayed in the main function to avoid duplication
LOCAL_DATASET_PATH_LISTING_JSON = "datasets/run_ready_904.json"
LOCAL_DATASET_PATH_LEGAL_JSONL = "datasets/legal_uk_greater_manchester.jsonl"


def rewrite_user_query(query: str) -> str:
    """Rewrite the user's query using a lightweight LLM for spelling correction.

    Falls back to the original query if the model or API key is unavailable.
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            return query
    except:
        return query

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=SecretStr(api_key))
        prompt = QUERY_REWRITING_PROMPT.format(query=query)
        result = llm.invoke(prompt)
        # Handle the result properly - it should be a ChatMessage with content
        if hasattr(result, "content"):
            return str(result.content).strip()
        elif isinstance(result, str):
            return result.strip()
        else:
            # Fallback: convert to string and strip
            return str(result).strip()
    except Exception:
        return query


class ChatbotWeb:
    """
    Main class that implements the RAG-based chatbot for real estate data.

    This class handles:
    1. Loading and processing CSV data from URLs
    2. Setting up vector stores for efficient retrieval
    3. Configuring the LLM and embedding models
    4. Managing the conversational interface
    5. Displaying source references for transparency
    """

    def __init__(self):
        """
        Initialize the ChatbotWeb instance.

        This method:
        1. Synchronizes Streamlit session state variables
        2. Configures the Language Model (LLM) based on user selection
        3. Sets up the embedding model for vector searches
        4. Creates chats folder for saving chat sessions
        """
        utils.sync_st_session()  # Ensure session state consistency
        self.llm = utils.configure_llm()  # Set up the language model (OpenAI or Llama)
        self.embedding_model = (
            utils.configure_embedding_model()
        )  # Set up embeddings for vector search

        # Create chats folder if it doesn't exist
        self.chats_folder = "chats"
        if not os.path.exists(self.chats_folder):
            os.makedirs(self.chats_folder)
            print(f"Created chats folder: {self.chats_folder}")

        # Initialize chat session tracking
        if "chat_session_id" not in st.session_state:
            st.session_state.chat_session_id = f"session_{int(time.time())}"
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

    @st.cache_resource(show_spinner="Creating Aspect-Based Vector Database", ttl=86400)
    def setup_vectordb(_self, local_file=None, jsonl_file=None, force_recreate=True):
        """
        Set up a vector database using the Aspect-Based chunking strategy.

        This method:
        1. Uses the Aspect-Based chunking strategy (best performer from evaluation)
        2. Creates separate chunks for crime, schools, transport, and overview aspects
        3. Generates embeddings for optimal retrieval
        4. Creates and returns a vector database for semantic search

        The function is cached using Streamlit's cache_resource for performance,
        with a time-to-live (TTL) of 24 hours before it needs to be refreshed.

        Parameters:
            _self: The ChatbotWeb instance
            local_file (str): Path to local JSON file (optional)
            jsonl_file (str): Path to JSONL file (optional)
            force_recreate (bool): If True, ignore existing database and create new one

        Returns:
            DocArrayInMemorySearch: A vector database for document retrieval
        """
        try:
            from aspect_based_chunker import create_aspect_based_vectordb
            import os

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
                )
                return None
            vectordb = create_aspect_based_vectordb(
                openai_api_key=openai_api_key,
                properties_file=LOCAL_DATASET_PATH_LISTING_JSON,
                legal_file=LOCAL_DATASET_PATH_LEGAL_JSONL,
                embedding_model=_self.embedding_model,
                force_recreate=force_recreate,
            )

            if vectordb:
                print("Aspect-Based Vector Database created successfully!")
                return vectordb
            else:
                st.error(
                    "Failed to create Aspect-Based Vector Database. Falling back to legacy method."
                )

        except Exception as e:
            print(f" Aspect-Based chunking failed: {e}")
            print(" Falling back to legacy chunking method...")
            st.warning(
                "Using legacy chunking method due to error in Aspect-Based chunking."
            )

    @st.cache_resource(show_spinner="Setting up RAG Pipeline", ttl=86400)
    def setup_rag_pipeline(_self):
        """
        Set up the clean RAG pipeline using the new RAGPipeline class.

        This method:
        1. Creates a clean RAG pipeline instance
        2. Handles all the complex setup internally
        3. Provides a simple interface for the UI

        Returns:
            RAGPipeline: The configured RAG pipeline instance
        """
        try:
            import os

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
                )
                return None

            # Create the clean RAG pipeline
            pipeline = create_rag_pipeline(openai_api_key)

            if pipeline is None:
                raise Exception("Failed to create RAG pipeline")

            print("RAG Pipeline setup completed successfully")
            return pipeline

        except Exception as e:
            st.error(f"Error setting up RAG pipeline: {e}")
            return None

    def save_chat_session(
        self, user_query: str, response: str, source_documents: Optional[list] = None
    ):
        """
        Save the current chat session to a JSON file

        Args:
            user_query: The user's question
            response: The AI's response
            source_documents: List of source documents used for the response
        """
        try:
            # Add message to session state
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "ai_response": response,
                "source_documents": [],
            }

            # Add source document information if available
            if source_documents:
                for doc in source_documents:
                    doc_info = {
                        "content": (
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        ),
                        "metadata": doc.metadata,
                        "type": (
                            doc.metadata.get("type", "unknown")
                            if hasattr(doc, "metadata")
                            else "unknown"
                        ),
                    }
                    message_data["source_documents"].append(doc_info)

            # Add to session messages
            st.session_state.chat_messages.append(message_data)

            # Save to JSON file
            session_id = st.session_state.chat_session_id
            filename = f"{session_id}.json"
            filepath = os.path.join(self.chats_folder, filename)

            # Prepare chat session data
            chat_session = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "total_messages": len(st.session_state.chat_messages),
                "messages": st.session_state.chat_messages,
                "app_version": "Aspect-Based Chunking v2.0",
                "chunking_strategy": "Aspect-Based (Best Performer)",
                "data_sources": [
                    LOCAL_DATASET_PATH_LISTING_JSON,
                    LOCAL_DATASET_PATH_LEGAL_JSONL,
                ],
            }

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chat_session, f, indent=2, ensure_ascii=False)

            print(f"Chat session saved: {filepath}")

        except Exception as e:
            print(f" Error saving chat session: {e}")
            traceback.print_exc()

    @utils.enable_chat_history  # Decorator to enable persistent chat history
    def main(self):
        """
        Main method to run the Streamlit application.
        4. Manages the chat interface
        5. Processes user queries and displays responses with citations

        """
        # Display the main header (moved here to avoid duplication)
        st.header("Chat with Real Estate AI Advisor")

        # Set up vector database and QA chain
        # Load local JSON datasets
        local_dataset_path = LOCAL_DATASET_PATH_LISTING_JSON
        jsonl_dataset_path = LOCAL_DATASET_PATH_LEGAL_JSONL

        # Add force recreate option in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Database Options**")
        force_recreate = st.sidebar.checkbox(
            " Force Recreate Database",
            value=False,
            help="Check this to ignore existing database and create new embeddings",
        )

        # Clear cache if force recreate is enabled
        if force_recreate:
            st.cache_resource.clear()
            st.info(" Cache cleared - will create new database")

        # Set up vector database with JSON datasets
        vectordb = self.setup_vectordb(
            local_file=local_dataset_path,
            jsonl_file=jsonl_dataset_path,
            force_recreate=force_recreate,
        )
        # Set up the RAG pipeline
        pipeline = self.setup_rag_pipeline()
        if pipeline is None:
            st.error("Failed to create RAG pipeline. Please check your configuration.")
            st.stop()

        # Show data sources info in compact sidebar format
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Data Sources**")
        st.sidebar.markdown("🧠 **Strategy:** Aspect-Based Chunking (Best Performer)")
        st.sidebar.markdown("**Storage:** Persistent (saved to disk)")
        st.sidebar.markdown("🤖 **Embeddings:** OpenAI (default)")
        st.sidebar.markdown(
            f"**Primary:** `{os.path.basename(LOCAL_DATASET_PATH_LISTING_JSON)}` (v2)"
        )
        st.sidebar.markdown(
            f"**Legal Data:** `{os.path.basename(LOCAL_DATASET_PATH_LEGAL_JSONL)}`"
        )

        # Show chat session info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💬 Chat Session**")
        st.sidebar.markdown(f"🆔 **Session ID:** `{st.session_state.chat_session_id}`")
        st.sidebar.markdown(f"**Messages:** {len(st.session_state.chat_messages)}")
        st.sidebar.markdown(
            f"**Saved to:** `chats/{st.session_state.chat_session_id}.json`"
        )

        # Display chat history
        if "messages" in st.session_state:
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Create chat input field
        user_query = st.chat_input(
            placeholder="Ask me about Manchester properties for sale!"
        )

        # Process query when user inputs a message
        if user_query:
            # Rewrite the query for spelling/grammar corrections
            corrected_query = rewrite_user_query(user_query)

            # Display the user message
            utils.display_msg(user_query, "user")

            # Display Advisor response with streaming
            with st.chat_message("Advisor"):
                # Show a subtle loading indicator
                with st.spinner("🤖 AI is thinking..."):
                    # Set up streaming handler to show response as it's generated
                    st_cb = StreamHandler(st.empty())

                    # Process the corrected query through the RAG pipeline
                    try:
                        # Use the pipeline's run_query method
                        result = pipeline.run_query(corrected_query)

                        # Convert to the format expected by the UI
                        ui_result = {"answer": result["answer"], "source_documents": []}

                        # Convert contexts back to document format for UI compatibility
                        for i, context in enumerate(result["contexts"]):
                            from langchain_core.documents.base import Document

                            doc = Document(
                                page_content=context,
                                metadata=(
                                    result["meta"].get("source_metadata", [{}])[i]
                                    if i
                                    < len(result["meta"].get("source_metadata", []))
                                    else {}
                                ),
                            )
                            ui_result["source_documents"].append(doc)

                        result = ui_result

                        # Debug logging
                        print(f"Query processed: {corrected_query}")
                        print(
                            f"Source documents retrieved: {len(result.get('source_documents', []))}"
                        )

                    except Exception as e:
                        print(f"Error in QA chain: {e}")
                        st.error(
                            f"Sorry, I encountered an error processing your query: {str(e)}"
                        )
                        return

                    # Extract and store the response
                    response = result["answer"]

                    # Add the response to session state for chat history
                    st.session_state.messages.append(
                        {"role": "Advisor", "content": response}
                    )

                    # Also add the user query to session state if not already there
                    if {
                        "role": "user",
                        "content": user_query,
                    } not in st.session_state.messages:
                        st.session_state.messages.append(
                            {"role": "user", "content": user_query}
                        )

                    utils.print_qa(ChatbotWeb, corrected_query, response)  # Log the Q&A

                    # Clear the streaming container and replace with properly formatted response
                    st_cb.container.empty()
                    st.write(response)

                    # Save chat session to JSON file
                    source_docs = result.get("source_documents", [])
                    self.save_chat_session(user_query, response, source_docs)

                    # Display source references horizontally with clipping and expandable view
                    if result["source_documents"]:
                        st.write("**References:**")

                        # Always show first 4 references horizontally
                        cols = st.columns(
                            min(len(result["source_documents"]), 4)
                        )  # Max 4 columns

                        for idx, doc in enumerate(result["source_documents"]):
                            if idx >= 4:  # Only show first 4 references initially
                                break

                            with cols[idx]:
                                # Extract source information based on metadata structure
                                src = (
                                    doc.metadata.get("source")
                                    or doc.metadata.get("property_url")
                                    or "unknown"
                                )
                                try:
                                    source_name = os.path.basename(src)
                                except Exception:
                                    source_name = str(src)

                                # Create a compact reference with clickable popup
                                ref_title = f"Ref {idx+1}: {source_name[:20]}{'...' if len(source_name) > 20 else ''}"

                                # Show document content in a popup when clicked
                                with st.popover(ref_title):
                                    st.caption("**Source:** " + str(src))
                                    st.text_area(
                                        "Content:",
                                        value=doc.page_content,
                                        height=150,
                                        disabled=True,
                                        key=f"ref_content_{idx}_1",
                                    )

                        # Show expandable section for additional references
                        if len(result["source_documents"]) > 4:
                            with st.expander(
                                f"📚 Show More References ({len(result['source_documents'])-4} more)",
                                expanded=False,
                            ):
                                # Show remaining references in a grid layout
                                remaining_docs = result["source_documents"][4:]

                                # Create rows of 4 columns for remaining references
                                for i in range(0, len(remaining_docs), 4):
                                    row_docs = remaining_docs[i : i + 4]
                                    row_cols = st.columns(len(row_docs))

                                    for j, doc in enumerate(row_docs):
                                        with row_cols[j]:
                                            # Extract source information
                                            src = (
                                                doc.metadata.get("source")
                                                or doc.metadata.get("property_url")
                                                or "unknown"
                                            )
                                            try:
                                                source_name = os.path.basename(src)
                                            except Exception:
                                                source_name = str(src)

                                            # Create reference with popup
                                            ref_title = f"Ref {i+j+5}: {source_name[:20]}{'...' if len(source_name) > 20 else ''}"

                                            with st.popover(ref_title):
                                                st.caption("**Source:** " + str(src))
                                                st.text_area(
                                                    "Content:",
                                                    value=doc.page_content,
                                                    height=150,
                                                    disabled=True,
                                                    key=f"ref_content_{i+j+5}_2",
                                                )

                            st.caption(
                                f"Showing 4 of {len(result['source_documents'])} references. Click 'Show More' to see all."
                            )
                        else:
                            st.caption(
                                f"Showing all {len(result['source_documents'])} references"
                            )


# Application entry point
if __name__ == "__main__":
    obj = ChatbotWeb()  # Create an instance of the chatbot
    obj.main()  # Run the main application loop
