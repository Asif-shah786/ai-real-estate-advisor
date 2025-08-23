from langchain_core.callbacks import BaseCallbackHandler
import re


class StreamHandler(BaseCallbackHandler):

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        # Format the text as it streams for better appearance
        formatted_text = self._format_streaming_text(self.text)
        self.container.markdown(formatted_text, unsafe_allow_html=True)

    def _format_streaming_text(self, text):
        """Format the streaming text to look better during generation"""
        # Remove any internal processing details that might appear
        text = re.sub(r'\{[^}]*"query"[^}]*\}', "", text)
        text = re.sub(r'\{[^}]*"filter"[^}]*\}', "", text)

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        # If text is empty after cleaning, provide a fallback
        if not text:
            text = "Generating response..."

        return text
