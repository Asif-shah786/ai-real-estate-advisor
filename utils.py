import os
import openai
from pydantic import SecretStr
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI as LangChainChatOpenAI

logger = get_logger("Langchain-Chatbot")


# decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # Initialize messages if not exists (but don't display them here)
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "Advisor",
                    "content": "How can I help you with finding real "
                    "estate properties?",
                }
            ]
        # Note: We don't display messages here to avoid duplicates
        # Messages are displayed in the main app loop

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/Advisor
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY",
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info(
            "Obtain your key from this link: https://platform.openai.com/account/api-keys"
        )
        st.stop()

    model = "gpt-4o"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [
            {"id": i.id, "created": datetime.fromtimestamp(i.created)}
            for i in client.models.list()
            if str(i.id).startswith("gpt")
        ]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model", options=available_models, key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()

    # Ensure we have valid values before returning
    if not model or not openai_api_key:
        st.error("Failed to get model or API key")
        st.stop()

    return model, openai_api_key


def configure_llm():
    llm = ChatOpenAI(
        model="gpt-4o",  # Upgraded from gpt-4o-mini to gpt-4o for better performance
        temperature=0.3,  # Balanced creativity and accuracy
        streaming=True,
        api_key=SecretStr(
            st.secrets["OPENAI_API_KEY"]
        ),  # Use SecretStr wrapper like app.py
        verbose=False,  # Suppress verbose output
    )
    return llm


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------" * 10
    logger.info(log_str.format(cls.__name__, question, answer))


@st.cache_resource
def configure_embedding_model():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=SecretStr(st.secrets["OPENAI_API_KEY"])
    )
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
