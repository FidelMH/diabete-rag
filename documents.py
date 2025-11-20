import os
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.indices.base import BaseIndex

class EmbedderRag:
    """
    Une classe pour encapsuler la configuration et la création 
    d'un VectorStoreIndex à partir de documents locaux.
    """
    def __init__(self, 
                 model_name: str = 'bge-m3', 
                 input_path: str = "./documents/2022_22.pdf",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        Initialise l'EmbedderRag avec les configurations nécessaires.

        Args:
            model_name (str): Le nom du modèle d'embedding Ollama.
            input_path (str): Le chemin vers le dossier ou le fichier à charger.
            chunk_size (int): La taille des morceaux de texte (chunks).
            chunk_overlap (int): Le chevauchement entre les morceaux de texte.
        """
        self.model_name = model_name
        self.input_path = input_path
        
        # Configuration du modèle d'embedding
        embed_model = OllamaEmbedding(
            model_name=self.model_name,
            ollama_additional_kwargs={
                "options": {
                    "num_gpu": 0,
                    "gpu_layers" : 0
                }
            }
        )
        
        # Application des paramètres globaux à LlamaIndex
        Settings.embed_model = embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        self.index: BaseIndex | None = None

    def build_index(self) -> BaseIndex:
        """
        Charge le(s) document(s) et construit l'index vectoriel.
        
        Returns:
            BaseIndex: L'index vectoriel créé.
        """
        print(f"Chargement du document depuis '{self.input_path}'...")
        
        if os.path.isdir(self.input_path):
            # Charge tous les fichiers d'un répertoire
            reader = SimpleDirectoryReader(input_dir=self.input_path)
        else:
            # Charge un fichier unique
            reader = SimpleDirectoryReader(input_files=[self.input_path])
        
        documents = reader.load_data()
        
        print("Création de l'index vectoriel...")
        self.index = VectorStoreIndex.from_documents(documents)
        print("Index créé avec succès.")
        
        return self.index

# --- Intégration pour la compatibilité ---
# Instancie la classe et construit l'index pour qu'il soit directement disponible
# comme la variable 'index' de l'ancien script, si d'autres modules l'importent.
rag_embedder_instance = EmbedderRag()
index = rag_embedder_instance.build_index()

if __name__ == '__main__':
    print("\nLe script documents.py a été exécuté directement.")
    print("L'index a été construit et est disponible via la variable 'index'.")
