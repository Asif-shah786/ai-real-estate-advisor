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
import requests  # For HTTP requests to load external data
import traceback  # For detailed error tracking
import validators  # For validating URLs
import streamlit as st  # Web UI framework
from streaming import StreamHandler  # Custom handler for streaming responses
from common.cfg import *  # Import configuration variables
from langchain.memory import (
    ConversationSummaryBufferMemory,
)  # For storing conversation context
from langchain.chains import ConversationalRetrievalChain  # Main RAG chain
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
from pydantic.v1 import SecretStr  # For handling sensitive API keys

# Configure the Streamlit page
st.set_page_config(
    page_title="🦾 AI Real Estate Advisor", page_icon="💬", layout="wide"
)
st.header("Chat with Real Estate AI Advisor")  # Main heading
LOCAL_DATASET_PATH = "dataset/structured_properties.csv"


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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=SecretStr(api_key))
        prompt = (
            "Rewrite the following user query, fixing spelling and grammar while keeping "
            "the original intent: "
            f"{query}"
        )
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
            print(f"✅ Created chats folder: {self.chats_folder}")

        # Initialize chat session tracking
        if "chat_session_id" not in st.session_state:
            st.session_state.chat_session_id = f"session_{int(time.time())}"
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

    @st.cache_resource(show_spinner="Creating Aspect-Based Vector Database", ttl=86400)
    def setup_vectordb(
        _self, websites=None, local_file=None, jsonl_file=None, force_recreate=False
    ):
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
            websites (list): List of URLs to CSV files (optional)
            local_file (str): Path to local CSV file (optional)
            jsonl_file (str): Path to JSONL file (optional, for backward compatibility)
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
                properties_file="dataset_v2/run_ready_100.json",
                legal_file="dataset_v2/legal_uk_greater_manchester.jsonl",
                embedding_model=_self.embedding_model,
                force_recreate=force_recreate,
            )

            if vectordb:
                print("✅ Aspect-Based Vector Database created successfully!")
                return vectordb
            else:
                st.error(
                    "Failed to create Aspect-Based Vector Database. Falling back to legacy method."
                )

        except Exception as e:
            print(f"⚠️ Aspect-Based chunking failed: {e}")
            print("🔄 Falling back to legacy chunking method...")
            st.warning(
                "Using legacy chunking method due to error in Aspect-Based chunking."
            )

    def setup_qa_chain(self, vectordb):
        """
                Set up a Conversational Retrieval Chain for question answering.

                This method configures:
                1. A retriever from the vector database with Maximum Marginal Relevance (MMR)
                2. Conversation memory to maintain context across interactions
                3. A QA chain that combines the retriever, memory, and LLM

        Parameters:
                    vectordb: The vector database to use for retrieval

        Returns:
                    ConversationalRetrievalChain: The configured QA chain
        """
        # Define retriever with cross-encoder reranking
        from retrieval import CrossEncoderRerankRetriever

        try:
            retriever = CrossEncoderRerankRetriever.from_vectorstore(
                vectordb, top_k=20, top_n=5
            )
        except Exception:
            # Fallback to simple similarity search if reranker cannot be initialized
            retriever = vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

        # Setup memory for contextual conversation using automatic summarization
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",  # Key used to access chat history in the chain
            output_key="answer",  # Key used to store the final answer
            return_messages=True,  # Return chat history as message objects
            max_token_limit=MEMORY_TOKEN_LIMIT,  # Summarize when token limit is reached
        )

        # Setup QA chain that combines the LLM, retriever, and memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,  # Language model configured in __init__
            retriever=retriever,  # Document retriever
            memory=memory,  # Conversation memory
            return_source_documents=True,  # Include source documents in the output
            verbose=False,  # Don't print debug info
        )
        return qa_chain

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
                    "dataset_v2/properties_with_crime_data.json",
                    "dataset_v2/legal_uk_greater_manchester.jsonl",
                ],
            }

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chat_session, f, indent=2, ensure_ascii=False)

            print(f"💾 Chat session saved: {filepath}")

        except Exception as e:
            print(f"⚠️ Error saving chat session: {e}")
            traceback.print_exc()

    @utils.enable_chat_history  # Decorator to enable persistent chat history
    def main(self):
        """
        Main method to run the Streamlit application.

        This method:
        1. Sets up the UI for URL input
        2. Handles adding and clearing data sources
        3. Creates the vector database and QA chain
        4. Manages the chat interface
        5. Processes user queries and displays responses with citations
        """
        csv_url = "CSV Data Set URL"  # Label for the URL input

        # Initialize session state for websites if not already set
        if "websites" not in st.session_state:
            st.session_state["websites"] = []  # List to store added URLs
            # Load default URLs from config
            st.session_state["value_urls"] = GIT_DATA_SET_URLS_STR.split("\n")

        # Set default URL for the input field
        url_val = ""
        value_urls = st.session_state.get("value_urls", [])
        if len(value_urls) >= 1:
            url_val = value_urls[0]  # Use first URL as default

        # Create text area for URL input in the sidebar
        web_url = st.sidebar.text_area(
            label=f"Enter {csv_url}s",
            placeholder="https://",
            # help="To add another website, modify this field after adding the website.",
            value=url_val,
        )
        # Alternative way to display URLs (commented out)
        # st.sidebar.text(GIT_DATA_SET_URLS_STR)

        # Button to add new URL to the list
        if st.sidebar.button(":heavy_plus_sign: Add Website"):
            # Validate URL format before adding
            valid_url = web_url.startswith("http") and validators.url(web_url)
            if not valid_url:
                # Show error for invalid URL
                st.sidebar.error(
                    f"Invalid URL! Please check {csv_url} that you have entered.",
                    icon="⚠️",
                )
            else:
                # Add valid URL to the session state
                st.session_state["websites"].append(web_url)

        # Button to clear all URLs
        if st.sidebar.button("Clear", type="primary"):
            st.session_state["websites"] = []

        # Button to start new chat session
        if st.sidebar.button("🆕 New Chat Session", type="secondary"):
            # Generate new session ID
            st.session_state.chat_session_id = f"session_{int(time.time())}"
            st.session_state.chat_messages = []
            st.session_state.messages = []
            st.rerun()

        # Remove duplicates by converting to set and back to list
        websites = list(set(st.session_state["websites"]))

        # Set up vector database and QA chain
        # First try to load local dataset, then add any URLs provided
        local_dataset_path = LOCAL_DATASET_PATH
        jsonl_dataset_path = "artifacts_v2/embedding_docs_v2.jsonl"

        # Add force recreate option in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Database Options**")
        force_recreate = st.sidebar.checkbox(
            "🔄 Force Recreate Database",
            value=False,
            help="Check this to ignore existing database and create new embeddings",
        )

        # Clear cache if force recreate is enabled
        if force_recreate:
            st.cache_resource.clear()
            st.info("🔄 Cache cleared - will create new database")

        # Set up vector database with JSONL dataset and any additional URLs
        vectordb = self.setup_vectordb(
            websites, local_dataset_path, jsonl_dataset_path, force_recreate
        )
        if vectordb is None:
            st.error(
                "Failed to create vector database. Please check your data sources."
            )
            st.stop()
        qa_chain = self.setup_qa_chain(vectordb)  # Configure the QA chain

        # Show data sources info in compact sidebar format
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Data Sources**")
        st.sidebar.markdown("🧠 **Strategy:** Aspect-Based Chunking (Best Performer)")
        st.sidebar.markdown("💾 **Storage:** Persistent (saved to disk)")
        st.sidebar.markdown("🤖 **Embeddings:** OpenAI (default)")
        st.sidebar.markdown(
            f"📄 **Primary:** `{os.path.basename(jsonl_dataset_path)}` (v2)"
        )
        st.sidebar.markdown(
            f"📁 **Fallback:** `{os.path.basename(LOCAL_DATASET_PATH)}`"
        )
        if websites:
            for url in websites:
                st.sidebar.markdown(
                    f"🌐 **External:** `{os.path.basename(url) if '/' in url else url}`"
                )
        else:
            st.sidebar.markdown("🌐 *No external URLs added*")

        # Show chat session info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💬 Chat Session**")
        st.sidebar.markdown(f"🆔 **Session ID:** `{st.session_state.chat_session_id}`")
        st.sidebar.markdown(f"💾 **Messages:** {len(st.session_state.chat_messages)}")
        st.sidebar.markdown(
            f"📁 **Saved to:** `chats/{st.session_state.chat_session_id}.json`"
        )

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
                # Set up streaming handler to show response as it's generated
                st_cb = StreamHandler(st.empty())

                # Process the corrected query through the QA chain
                result = qa_chain.invoke(
                    {"question": corrected_query},
                    {"callbacks": [st_cb]},  # Use streaming callback
                )

                # Extract and store the response
                response = result["answer"]
                st.session_state.messages.append(
                    {"role": "Advisor", "content": response}
                )
                utils.print_qa(ChatbotWeb, corrected_query, response)  # Log the Q&A

                # Save chat session to JSON file
                source_docs = result.get("source_documents", [])
                self.save_chat_session(user_query, response, source_docs)

                # Display source references for transparency
                for idx, doc in enumerate(result["source_documents"], 1):
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

                    # Create a reference title with clickable popup
                    ref_title = f":blue[Reference {idx}: *{source_name}*]"
                    # Show document content in a popup when clicked
                    with st.popover(ref_title):
                        st.caption(doc.page_content)


# Application entry point
if __name__ == "__main__":
    obj = ChatbotWeb()  # Create an instance of the chatbot
    obj.main()  # Run the main application loop
