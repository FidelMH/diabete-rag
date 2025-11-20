import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import Settings

# Charge les variables d'environnement à partir d'un fichier .env
load_dotenv()

# Récupère la clé API Groq. LlamaIndex peut le faire automatiquement, 
# mais cette vérification explicite est plus claire.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("La clé API Groq ('GROQ_API_KEY') n'a pas été trouvée. "
                     "Veuillez la définir dans un fichier .env à la racine de votre projet.")

# Définition du modèle de langage (LLM) via l'API Groq.
llm = Groq(model="openai/gpt-oss-20b", api_key=api_key)

# Appliquer ce LLM à la configuration globale de LlamaIndex
Settings.llm = llm

print("LLM configuré avec l'API Groq (modèle 'openai/gpt-oss-20b').")
