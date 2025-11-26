# Diabète RAG

RAG (Retrieval Augmented Generation) system for answering questions about diabetes from medical documents.

## 🚀 Quick Start

### 1. Installation

```bash
git clone [<repository-url>](https://github.com/FidelMH/diabete-rag)
cd diabete-rag
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env`:

```bash
# Windows
copy .env.example .env

# Linux/MacOS
cp .env.example .env
```

Edit `.env` and configure your settings. Choose one of the following:

**Option A: Azure OpenAI (Recommended for production)**
```env
# Embeddings
EMBEDDING_PROVIDER=azure
EMBEDDING_API_KEY=your_azure_key
EMBEDDING_API_BASE=https://your-resource.openai.azure.com/openai/v1/
EMBEDDING_MODEL_NAME=text-embedding-3-large

# LLM
LLM_API_KEY=your_azure_key
LLM_API_BASE=https://your-resource.openai.azure.com/openai/v1/
LLM_MODEL_NAME=gpt-4
```

**Option B: Local Embeddings + Azure LLM (Development/Cost optimization)**
```env
# Embeddings (local)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu

# LLM (Azure)
LLM_API_KEY=your_azure_key
LLM_API_BASE=https://your-resource.openai.azure.com/openai/v1/
LLM_MODEL_NAME=gpt-4
```

**For local embeddings, install additional dependencies:**
```bash
pip install sentence-transformers torch
```

### 3. Add Documents

```bash
mkdir documents
# Place your PDF files in the documents/ folder
```

### 4. Run the Application

**Web Interface (Streamlit):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python main.py
```

## 🧪 Quick Test

Test the system with a simple question:

```bash
python -c "
from documents import EmbedderRag
from llm import LlmManager

# Initialize
LlmManager()
embedder = EmbedderRag(input_path='./documents')
index = embedder.build_or_load_index()

# Query
engine = index.as_query_engine(similarity_top_k=5)
response = engine.query('What is diabetes?')
print(response)
"
```

## ⚙️ Configuration Guide

### Embedding Providers

#### Azure OpenAI (Cloud)
- Production-ready, scalable
- Requires Azure subscription
- Best quality for multilingual content
- Models: `text-embedding-3-large`, `text-embedding-3-small`

#### HuggingFace (Local)
- Free, runs on your machine
- No API costs
- Slower on CPU, fast on GPU
- Recommended models:
  - `BAAI/bge-large-en-v1.5` - Best quality (English)
  - `sentence-transformers/all-MiniLM-L6-v2` - Fast, good quality
  - `intfloat/multilingual-e5-large` - Multilingual support

### LLM Configuration

Only Azure OpenAI is supported for the LLM.

**Temperature settings:**
- `0.1-0.3`: Highly factual medical responses (recommended)
- `0.5-0.7`: Balanced responses
- `0.8-1.0`: More creative, conversational

Modify in your code:
```python
LlmManager(temperature=0.3)  # More factual
```

### Document Processing

The system includes automatic PDF cleaning:

```python
embedder = EmbedderRag(
    input_path="./documents",
    clean_text=True,           # Remove headers, footers, page numbers
    remove_urls=True,          # Remove URLs and emails
    normalize_medical=False    # Normalize medical abbreviations
)
```

Preview document statistics before indexing:
```bash
python preprocess_pdfs.py --preview
```

## 📁 Project Structure

```
diabete-rag/
├── documents/              # Your PDF documents
├── storage/               # Vector index (auto-created)
├── app.py                 # Streamlit web interface
├── main.py                # CLI interface
├── embedding_manager.py   # Universal embedding configuration
├── llm.py                 # LLM configuration
├── documents.py           # Document indexing with cleaning
├── text_cleaner.py        # PDF preprocessing
├── preprocess_pdfs.py     # Document analysis tool
├── test_ragas.py          # RAG evaluation with RAGAS
└── requirements.txt       # Dependencies
```

## 📊 Evaluation

Evaluate RAG quality with RAGAS metrics:

```bash
python test_ragas.py
```

**Metrics:**
- **Faithfulness**: Answer accuracy vs source documents
- **Answer Relevancy**: Response relevance to question
- **Context Precision**: Retrieved context accuracy
- **Context Recall**: Coverage of relevant information

Results saved to `ragas_evaluation_results.csv`

## 🚀 Azure Deployment

See [DEPLOY_AZURE.md](DEPLOY_AZURE.md) for Azure App Service deployment instructions.

## ❓ Troubleshooting

### "EMBEDDING_PROVIDER not found"
Ensure `.env` exists and contains `EMBEDDING_PROVIDER=azure` or `local`

### "EMBEDDING_API_KEY not found" or "LLM_API_KEY not found"
Check that all required environment variables are set in `.env` based on your chosen provider

### "No module named 'sentence_transformers'"
For local embeddings: `pip install sentence-transformers torch`

### Local embeddings are slow
- Set `EMBEDDING_DEVICE=cuda` if you have a GPU
- Or switch to a smaller model: `sentence-transformers/all-MiniLM-L6-v2`

### Poor response quality
- Run `python preprocess_pdfs.py --preview` to check document quality
- Enable `clean_text=True` in document processing
- Lower LLM temperature for more factual responses: `LlmManager(temperature=0.3)`

### Streamlit app won't start
```bash
pip install streamlit
streamlit run app.py --logger.level=debug
```

## 📋 Features

### PDF Cleaning
The system automatically cleans PDF documents for better RAG quality:

- ✅ **Headers/footers** - Automatic detection and removal
- ✅ **Page numbers** - Removal of page numbers
- ✅ **Special characters** - Normalization of spaces, apostrophes
- ✅ **Hyphenated words** - Correction of line-break hyphenation
- ✅ **URLs and emails** - Optional removal
- ✅ **Medical terms** - Optional abbreviation normalization

---

**Medical Disclaimer**: This chatbot is for informational purposes only and does not constitute medical advice. Consult a qualified healthcare professional for medical questions.
