import os
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

class EmbeddingManager:
    """
    Gère la configuration et l'initialisation du modèle d'embedding de manière générique.
    """
    def __init__(self):
        """
        Initialise le EmbeddingManager en chargeant la configuration depuis le .env
        et en configurant le modèle d'embedding.
        """
        load_dotenv()
        
        # Charger les variables d'environnement pour l'embedding
        api_base = os.getenv("API_BASE_URL")
        api_key = os.getenv("API_KEY")
        model_name = os.getenv("EMBEDDING_MODEL_NAME")

        if not api_base:
            raise ValueError("API_BASE_URL non trouvée dans le fichier .env")
        if not api_key:
            raise ValueError("La clé API_KEY n'est pas définie dans le fichier .env")
        if not model_name:
            raise ValueError("EMBEDDING_MODEL_NAME n'est pas définie dans le fichier .env")

        print(f"Configuration du modèle d'embedding '{model_name}' via l'API '{api_base}'...")
        
        # Utilisation de OpenAIEmbedding qui est générique et peut pointer vers n'importe quelle API
        # compatible OpenAI (y compris Ollama via son endpoint /v1)
        embed_model = OpenAIEmbedding(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
        )
        
        # Appliquer ce modèle d'embedding à la configuration globale de LlamaIndex
        Settings.embed_model = embed_model
        
        print("✓ Modèle d'embedding configuré avec succès.")
        
        # Sauvegarder les paramètres pour une utilisation ultérieure (ex: RAGAS)
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key

