import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

class LlmManager:
    """
    LLM manager for Azure OpenAI (or OpenAI-compatible APIs).

    Note: This project uses Azure OpenAI exclusively for LLM.
    For embeddings, see EmbeddingManager which supports both cloud and local.
    """

    def __init__(self, temperature: float = 0.7):
        """
        Initialize LLM with Azure OpenAI configuration.

        Args:
            temperature: Sampling temperature (0.0-1.0)
                - Low (0.1-0.5): More factual, deterministic responses
                - High (0.7-1.0): More creative, diverse responses
        """
        load_dotenv()

        api_key = os.getenv("LLM_API_KEY")
        api_base = os.getenv("LLM_API_BASE")
        model_name = os.getenv("LLM_MODEL_NAME")

        if not api_key:
            raise ValueError("LLM_API_KEY not found in .env")
        if not api_base:
            raise ValueError("LLM_API_BASE not found in .env")
        if not model_name:
            raise ValueError("LLM_MODEL_NAME not found in .env")

        print(f"Configuring LLM...")
        print(f"  Model: {model_name}")
        print(f"  Temperature: {temperature}")
        print(f"  API: {api_base}")

        self.llm = OpenAILike(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=True,
            temperature=temperature
        )

        # Apply to global settings
        Settings.llm = self.llm
        print("✓ LLM configured successfully.")

    def get_config(self) -> dict:
        """Return LLM configuration for downstream use."""
        return {
            "model_name": os.getenv("LLM_MODEL_NAME"),
            "api_key": os.getenv("LLM_API_KEY"),
            "api_base": os.getenv("LLM_API_BASE"),
        }
