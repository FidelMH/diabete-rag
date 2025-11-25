# Diabète RAG

Système RAG (Retrieval Augmented Generation) pour répondre aux questions sur le diabète à partir de documents médicaux.

## Installation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Installer Ollama et télécharger le modèle d'embedding
```bash
ollama pull bge-m3
```

### 3. Préparer les documents
Mettre les documents PDF dans un dossier `documents/` à la racine du projet.

### 4. Configuration des variables d'environnement
Copier le fichier `.env.example` et le renommer en `.env`, puis ajouter votre clé API (Groq, OpenAI, etc.):
```
API_KEY=votre_clé_api_ici
```

## Utilisation

### Lancer l'application Streamlit (Interface Web)
```bash
streamlit run app.py
```
L'application se lancera dans votre navigateur. Vous pourrez :
- Poser des questions sur le diabète via une interface de chat
- Voir l'historique de conversation
- Consulter les sources utilisées pour générer les réponses

### Utiliser la ligne de commande (CLI)
```bash
python main.py
```

### Créer l'index vectoriel (programmation)
```python
from documents import EmbedderRag

embedder = EmbedderRag(
    model_name="bge-m3",
    input_path="./documents"
)
index = embedder.build_or_load_index()
```

### Évaluer le système RAG avec RAGAS
```bash
python test_ragas.py
```

Cette commande va :
1. Générer automatiquement des questions de test à partir de vos documents
2. Évaluer votre système RAG avec 4 métriques :
   - **Faithfulness** : Fidélité de la réponse aux documents
   - **Answer Relevancy** : Pertinence de la réponse
   - **Context Precision** : Précision du contexte récupéré
   - **Context Recall** : Rappel du contexte pertinent
3. Sauvegarder les résultats dans `ragas_evaluation_results.csv`

## Structure du projet

```
diabete-rag/
├── documents/           # Documents PDF à indexer
├── storage/            # Index vectoriel persistant
├── app.py              # Application Streamlit (Frontend Web)
├── main.py             # Interface CLI
├── documents.py        # Classe pour créer l'index
├── llm.py              # Gestionnaire du LLM
├── test_ragas.py       # Évaluation avec RAGAS
├── requirements.txt    # Dépendances Python
└── .env               # Variables d'environnement (clé API)
```