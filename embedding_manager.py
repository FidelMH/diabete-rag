import os
from dotenv import load_dotenv
from llama_index.core import Settings

class EmbeddingManager:
    """
    Universal embedding manager supporting both cloud and local embeddings.

    Supported providers:
    - azure: Azure OpenAI embeddings (OpenAI-compatible API)
    - local: HuggingFace embeddings (sentence-transformers)
    """

    SUPPORTED_PROVIDERS = ["azure", "local"]

    def __init__(self):
        """
        Initialize EmbeddingManager with provider-agnostic configuration.
        Automatically detects provider from environment and configures accordingly.
        """
        load_dotenv()

        # Get provider type
        provider = os.getenv("EMBEDDING_PROVIDER", "azure").lower()

        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {self.SUPPORTED_PROVIDERS}, "
                f"got '{provider}'"
            )

        print(f"Configuring '{provider}' embedding provider...")

        # Route to appropriate configuration method
        if provider == "azure":
            embed_model = self._configure_azure_embeddings()
        elif provider == "local":
            embed_model = self._configure_local_embeddings()

        # Apply to global LlamaIndex settings
        Settings.embed_model = embed_model
        print("✓ Embedding model configured successfully.")

        # Save configuration for downstream use (e.g., RAGAS)
        self.provider = provider
        self.embed_model = embed_model

    def _configure_azure_embeddings(self):
        """Configure Azure OpenAI embeddings (or any OpenAI-compatible API)."""
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.getenv("EMBEDDING_API_KEY")
        api_base = os.getenv("EMBEDDING_API_BASE")
        model_name = os.getenv("EMBEDDING_MODEL_NAME")

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY not found in .env")
        if not api_base:
            raise ValueError("EMBEDDING_API_BASE not found in .env")
        if not model_name:
            raise ValueError("EMBEDDING_MODEL_NAME not found in .env")

        print(f"  Model: {model_name}")
        print(f"  API: {api_base}")

        return OpenAIEmbedding(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
        )

    def _configure_local_embeddings(self):
        """Configure local HuggingFace embeddings."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Max length: {max_length}")
        print("  Note: First run will download the model (~500MB)")
        print("  Loading HuggingFace model... This may take a moment.")

        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            max_length=max_length,
        )

        print("  ✓ HuggingFace model loaded successfully.")
        return embed_model

    def get_config(self) -> dict:
        """
        Return configuration dict for use in evaluation tools (RAGAS, etc.).
        Useful for passing embedding config to other libraries.
        """
        if self.provider == "azure":
            return {
                "provider": "azure",
                "model_name": os.getenv("EMBEDDING_MODEL_NAME"),
                "api_key": os.getenv("EMBEDDING_API_KEY"),
                "api_base": os.getenv("EMBEDDING_API_BASE"),
            }
        else:  # local
            return {
                "provider": "local",
                "model_name": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5"),
                "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
            }
