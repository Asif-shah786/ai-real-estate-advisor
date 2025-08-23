"""
Streaming handler for real-time response display.

This module provides a StreamHandler class that enables streaming
responses in the Streamlit UI for better user experience.
"""

import streamlit as st
from typing import Any, Optional
from langchain_core.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    """
    Stream handler for displaying streaming responses in Streamlit.

    This class handles the streaming of LLM responses to provide
    real-time feedback to users during generation.
    """

    def __init__(self, container: Any, initial_text: str = ""):
        """
        Initialize the stream handler.

        Args:
            container: Streamlit container to display the stream
            initial_text: Initial text to display
        """
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Handle new tokens from the LLM.

        Args:
            token: New token to add to the stream
            **kwargs: Additional keyword arguments
        """
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """
        Handle the end of LLM generation.

        Args:
            response: Final response from the LLM
            **kwargs: Additional keyword arguments
        """
        # Remove the cursor and display final text
        self.container.markdown(self.text)

    def on_llm_error(self, error: str, **kwargs) -> None:
        """
        Handle LLM errors.

        Args:
            error: Error message
            **kwargs: Additional keyword arguments
        """
        self.container.error(f"Error: {error}")

    def get_text(self) -> str:
        """
        Get the accumulated text.

        Returns:
            Accumulated text from the stream
        """
        return self.text


class SimpleStreamHandler:
    """
    Simple stream handler for basic streaming functionality.

    This is a simplified version that can be used when
    the full LangChain callback system is not needed.
    """

    def __init__(self, container: Any):
        """
        Initialize the simple stream handler.

        Args:
            container: Streamlit container to display the stream
        """
        self.container = container
        self.text = ""

    def add_token(self, token: str):
        """
        Add a token to the stream.

        Args:
            token: Token to add
        """
        self.text += token
        self.container.markdown(self.text + "▌")

    def finish(self):
        """Finish the stream and display final text."""
        self.container.markdown(self.text)

    def get_text(self) -> str:
        """
        Get the accumulated text.

        Returns:
            Accumulated text from the stream
        """
        return self.text


def create_stream_handler(container: Any, handler_type: str = "langchain") -> Any:
    """
    Factory function to create appropriate stream handler.

    Args:
        container: Streamlit container to display the stream
        handler_type: Type of handler to create ("langchain" or "simple")

    Returns:
        Stream handler instance
    """
    if handler_type == "langchain":
        return StreamHandler(container)
    elif handler_type == "simple":
        return SimpleStreamHandler(container)
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")
