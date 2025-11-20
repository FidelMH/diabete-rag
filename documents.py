import os
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.indices.base import BaseIndex

class EmbedderRag:
    """
    Classe modifiée pour gérer la persistance de l'index.
    """
    def __init__(self, 
                 model_name: str = 'bge-m3', 
                 input_path: str = "./documents",
                 persist_dir: str = "./storage",  # Dossier pour la persistance
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        Initialise l'EmbedderRag avec les configurations nécessaires.
        """
        self.model_name = model_name
        self.input_path = input_path
        self.persist_dir = persist_dir
        
        # Configuration du modèle d'embedding
        embed_model = OllamaEmbedding(
            model_name=self.model_name,
            ollama_additional_kwargs={"options": {"num_gpu": 0, "gpu_layers": 0}}
        )
        
        # Application des paramètres globaux à LlamaIndex
        Settings.embed_model = embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        self.index: BaseIndex | None = None

    def build_or_load_index(self) -> BaseIndex:
        """
        Charge l'index depuis le disque s'il existe, sinon le construit et le sauvegarde.
        """
        if not os.path.exists(self.persist_dir):
            # CAS 1: L'index n'existe pas, on le construit
            print(f"Chargement du document depuis '{self.input_path}'...")
            if os.path.isdir(self.input_path):
                reader = SimpleDirectoryReader(input_dir=self.input_path)
            else:
                reader = SimpleDirectoryReader(input_files=[self.input_path])
            documents = reader.load_data()
            
            print("Création et persistance de l'index (cela peut prendre du temps)...")
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"Index sauvegardé dans '{self.persist_dir}'.")
        else:
            # CAS 2: L'index existe, on le charge
            print(f"Chargement de l'index depuis '{self.persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("Index chargé avec succès.")
            
        return self.index

# --- Intégration et exécution ---
# Instancie la classe et charge ou construit l'index.
rag_embedder_instance = EmbedderRag()
index = rag_embedder_instance.build_or_load_index()

if __name__ == '__main__':
    print("\nLe script documents.py a été exécuté directement.")
    print("L'index est prêt à l'emploi.")