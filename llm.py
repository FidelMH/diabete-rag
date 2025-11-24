import os
from dotenv import load_dotenv
# from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

class LlmManager:
    """
    Gère la configuration et l'initialisation du modèle de langage (LLM).
    """
    def __init__(self):
        """
        Initialise le LlmManager en chargeant la clé API et en configurant le LLM.

        Args:
            model_name (str): Le nom du modèle Groq à utiliser.
        """
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        API_BASE_URL = os.getenv("API_BASE_URL")
        LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

        if not API_KEY:
            raise ValueError("La clé API_KEY n'est pas définie dans le fichier .env")
        if not API_BASE_URL:
            raise ValueError("API_BASE_URL non trouvée dans le fichier .env")
        if not LLM_MODEL_NAME:
            raise ValueError("LLM_MODEL_NAME n'est pas définie dans le fichier .env")

        print(f"Configuration du LLM avec l'API  (modèle '{LLM_MODEL_NAME}')...")
        self.llm = OpenAILike(model=LLM_MODEL_NAME, api_key=API_KEY, api_base=API_BASE_URL,is_chat_model=True,temperature=1.0)
        
        # Appliquer ce LLM à la configuration globale pour qu'il soit utilisé partout
        Settings.llm = self.llm
        print("LLM configuré avec succès.")