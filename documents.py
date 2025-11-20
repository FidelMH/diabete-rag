from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.indices.base import BaseIndex

class EmbedderRag:
    """
    Une classe pour encapsuler la configuration et la création 
    d'un VectorStoreIndex à partir de documents locaux.
    """
    def __init__(self, 
                 model_name: str = 'royalpha/sentence-camembert-large', 
                 directory_path: str = "./documents",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        Initialise l'EmbedderRag avec les configurations nécessaires.

        Args:
            model_name (str): Le nom du modèle d'embedding Ollama.
            directory_path (str): Le chemin vers le dossier contenant les documents.
            chunk_size (int): La taille des morceaux de texte (chunks).
            chunk_overlap (int): Le chevauchement entre les morceaux de texte.
        """
        self.model_name = model_name
        self.directory_path = directory_path
        
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
        Charge les documents et construit l'index vectoriel.
        
        Returns:
            BaseIndex: L'index vectoriel créé.
        """
        print(f"Chargement des documents depuis '{self.directory_path}'...")
        documents = SimpleDirectoryReader(self.directory_path).load_data()
        
        print("Création de l'index vectoriel...")
        self.index = VectorStoreIndex.from_documents(documents)
        print("Index créé avec succès.")
        
        return self.index

