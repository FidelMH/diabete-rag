import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import Settings

class LlmManager:
    """
    Gère la configuration et l'initialisation du modèle de langage (LLM).
    """
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        """
        Initialise le LlmManager en chargeant la clé API et en configurant le LLM.

        Args:
            model_name (str): Le nom du modèle Groq à utiliser.
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Clé API Groq ('GROQ_API_KEY') non trouvée dans le fichier .env")

        print(f"Configuration du LLM avec l'API Groq (modèle '{model_name}')...")
        self.llm = Groq(model=model_name, api_key=api_key)
        
        # Appliquer ce LLM à la configuration globale pour qu'il soit utilisé partout
        Settings.llm = self.llm
        print("LLM configuré avec succès.")