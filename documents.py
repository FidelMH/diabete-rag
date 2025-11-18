# from llama_index.llms.ollama import Ollama 
# from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

embed_model = OllamaEmbedding(
    model_name='royalpha/sentence-camembert-large',
    ollama_additional_kwargs={
        "options": {
            "num_gpu": 0,
            "gpu_layers" : 0
        }
    }
)

Settings.embed_model = embed_model
Settings.chunk_size = 800
Settings.chunk_overlap = 100

documents = SimpleDirectoryReader("./documents").load_data()
index = VectorStoreIndex.from_documents(documents)

