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
from text_cleaner import TextCleaner, get_document_stats

class EmbedderRag:
    """
    Classe modifiée pour gérer la persistance de l'index avec nettoyage des documents.
    """
    def __init__(self,
                 model_name: str = 'bge-m3',
                 input_path: str = "./documents",
                 persist_dir: str = "./storage",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 clean_text: bool = True,
                 remove_urls: bool = True,
                 normalize_medical: bool = False):
        """
        Initialise l'EmbedderRag avec les configurations nécessaires.

        Args:
            model_name: Nom du modèle d'embedding Ollama
            input_path: Chemin vers les documents
            persist_dir: Dossier pour la persistance de l'index
            chunk_size: Taille des chunks pour le découpage
            chunk_overlap: Chevauchement entre les chunks
            clean_text: Si True, nettoie le texte des documents
            remove_urls: Si True, supprime les URLs et emails
            normalize_medical: Si True, normalise les termes médicaux
        """
        self.model_name = model_name
        self.input_path = input_path
        self.persist_dir = persist_dir
        self.clean_text = clean_text
        self.remove_urls = remove_urls
        self.normalize_medical = normalize_medical
        
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
            print(f"Chargement des documents depuis '{self.input_path}'...")
            if os.path.isdir(self.input_path):
                reader = SimpleDirectoryReader(input_dir=self.input_path)
            else:
                reader = SimpleDirectoryReader(input_files=[self.input_path])
            documents = reader.load_data()

            print(f"✓ {len(documents)} document(s) chargé(s)")

            # Afficher les statistiques avant nettoyage
            if self.clean_text:
                print("\n📊 Statistiques avant nettoyage:")
                stats_before = get_document_stats(documents)
                for key, value in stats_before.items():
                    print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

                print("\n🧹 Nettoyage des documents en cours...")
                documents = TextCleaner.clean_documents(
                    documents,
                    remove_urls=self.remove_urls,
                    normalize_medical=self.normalize_medical
                )

                print("✓ Nettoyage terminé!")

                # Afficher les statistiques après nettoyage
                print("\n📊 Statistiques après nettoyage:")
                stats_after = get_document_stats(documents)
                for key, value in stats_after.items():
                    print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

                # Calcul de la réduction
                reduction_chars = ((stats_before['total_characters'] - stats_after['total_characters']) /
                                 stats_before['total_characters'] * 100)
                print(f"\n   Réduction du texte: {reduction_chars:.1f}%")

            print("\n🔨 Création de l'index vectoriel (cela peut prendre du temps)...")
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"✓ Index sauvegardé dans '{self.persist_dir}'.")
        else:
            # CAS 2: L'index existe, on le charge
            print(f"Chargement de l'index depuis '{self.persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("✓ Index chargé avec succès.")

        return self.index