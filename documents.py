import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.indices.base import BaseIndex
from text_cleaner import TextCleaner, get_document_stats
from embedding_manager import EmbeddingManager

class EmbedderRag:
    """
    Modified class to manage index persistence with document cleaning.
    """
    def __init__(self,
                 input_path: str = "./documents",
                 persist_dir: str = "./storage",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 clean_text: bool = True,
                 remove_urls: bool = True,
                 normalize_medical: bool = False):
        """
        Initialize EmbedderRag with necessary configurations.

        Args:
            input_path: Path to documents
            persist_dir: Directory for index persistence
            chunk_size: Chunk size for splitting
            chunk_overlap: Overlap between chunks
            clean_text: If True, clean document text
            remove_urls: If True, remove URLs and emails
            normalize_medical: If True, normalize medical terms
        """
        self.input_path = input_path
        self.persist_dir = persist_dir
        self.clean_text = clean_text
        self.remove_urls = remove_urls
        self.normalize_medical = normalize_medical

        # Configure embedding model via manager
        self.embedding_manager = EmbeddingManager()

        # Application des paramètres globaux à LlamaIndex
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        self.index: BaseIndex | None = None

    def build_or_load_index(self) -> BaseIndex:
        """
        Load index from disk if it exists, otherwise build and save it.
        """
        if not os.path.exists(self.persist_dir):
            # CASE 1: Index doesn't exist, build it
            print(f"Loading documents from '{self.input_path}'...")
            if os.path.isdir(self.input_path):
                reader = SimpleDirectoryReader(input_dir=self.input_path)
            else:
                reader = SimpleDirectoryReader(input_files=[self.input_path])
            documents = reader.load_data()

            print(f"✓ {len(documents)} document(s) loaded")

            # Display statistics before cleaning
            if self.clean_text:
                print("\n📊 Statistics before cleaning:")
                stats_before = get_document_stats(documents)
                for key, value in stats_before.items():
                    print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

                print("\n🧹 Cleaning documents...")
                documents = TextCleaner.clean_documents(
                    documents,
                    remove_urls=self.remove_urls,
                    normalize_medical=self.normalize_medical
                )

                print("✓ Cleaning completed!")

                # Display statistics after cleaning
                print("\n📊 Statistics after cleaning:")
                stats_after = get_document_stats(documents)
                for key, value in stats_after.items():
                    print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

                # Calculate reduction
                reduction_chars = ((stats_before['total_characters'] - stats_after['total_characters']) /
                                 stats_before['total_characters'] * 100)
                print(f"\n   Text reduction: {reduction_chars:.1f}%")

            print("\n🔨 Creating vector index (this may take some time)...")
            provider = self.embedding_manager.provider
            if provider == "local":
                print(f"   Using local embeddings: {self.embedding_manager.embed_model.model_name}")
                print(f"   Processing {len(documents)} document(s)...")
                print(f"   This will generate embeddings for each text chunk.")
            else:
                print(f"   Using Azure OpenAI embeddings")
                print(f"   Processing {len(documents)} document(s)...")

            self.index = VectorStoreIndex.from_documents(documents, show_progress=True)

            print(f"\n💾 Saving index to '{self.persist_dir}'...")
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"✓ Index saved successfully.")
        else:
            # CASE 2: Index exists, load it
            print(f"Loading index from '{self.persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("✓ Index loaded successfully.")

        return self.index