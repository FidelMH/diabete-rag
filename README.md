# Diabète RAG

Système RAG (Retrieval Augmented Generation) pour répondre aux questions sur le diabète à partir de documents médicaux.

## 🚀 Démarrage Rapide

```bash
# 1. Cloner et installer
git clone <repository-url>
cd diabete-rag
pip install -r requirements.txt

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditez .env avec votre clé API

# 3. Ajouter vos documents PDF
mkdir documents
# Placez vos PDFs dans le dossier documents/

# 4. Lancer l'application
streamlit run app.py
```

Votre application web s'ouvrira automatiquement dans votre navigateur !

## Installation

### 1. Cloner le projet
```bash
git clone <repository-url>
cd diabete-rag
```

### 2. Créer un environnement virtuel (recommandé)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration des variables d'environnement
Copier le fichier [.env.example](.env.example) et le renommer en `.env`:
```bash
# Windows
copy .env.example .env

# Linux/MacOS
cp .env.example .env
```

Ensuite, éditez le fichier `.env` et choisissez l'une des configurations suivantes:

#### Option A : Azure OpenAI (Recommandé pour la production)
```env
API_KEY=your_azure_api_key_here
API_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
LLM_MODEL_NAME=gpt-4
EMBEDDING_MODEL_NAME=text-embedding-3-large
```

#### Option B : Groq (Rapide et gratuit)
```env
API_KEY=gsk_your_groq_api_key_here
API_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL_NAME=llama-3.1-70b-versatile
EMBEDDING_MODEL_NAME=bge-m3
```

### 5. Préparer les documents
Créer un dossier `documents/` à la racine du projet et y placer vos documents PDF:
```bash
mkdir documents
# Copier vos fichiers PDF dans ce dossier
```

**Optionnel :** Analyser les documents avant indexation
```bash
python preprocess_pdfs.py --preview
```
Cette commande affiche des statistiques sur vos documents, détecte les problèmes potentiels et recommande les meilleures options de nettoyage.

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
    input_path="./documents",
    clean_text=True,           # Active le nettoyage automatique des PDFs
    remove_urls=True,          # Supprime les URLs et emails
    normalize_medical=False    # Normalise les termes médicaux (optionnel)
)
index = embedder.build_or_load_index()
```

**Options de nettoyage disponibles :**
- `clean_text=True` : Active le nettoyage automatique (recommandé)
  - Suppression des headers/footers répétitifs
  - Normalisation des espaces et sauts de ligne
  - Suppression des numéros de page
  - Correction des mots coupés (hyphenation)
  - Nettoyage des caractères spéciaux
- `remove_urls=True` : Supprime les URLs et emails
- `normalize_medical=True` : Normalise les abréviations médicales (DT1 → diabète de type 1, etc.)

### Analyser et prétraiter les documents
```bash
python preprocess_pdfs.py --preview
```

Cette commande permet de :
- Afficher des statistiques sur les documents (nombre de mots, caractères, etc.)
- Comparer avant/après le nettoyage
- Détecter les problèmes potentiels (lignes répétitives, URLs, etc.)
- Obtenir des recommandations sur les options de nettoyage à utiliser

Options disponibles :
- `--path ./documents` : Spécifier le dossier des documents
- `--preview` : Afficher un aperçu détaillé de chaque document

### Évaluer le système RAG avec RAGAS
```bash
python test_ragas.py
```

Cette commande va :
1. Nettoyer automatiquement les documents (résout les erreurs headlines)
2. Générer automatiquement des questions de test à partir de vos documents
3. Évaluer votre système RAG avec 4 métriques :
   - **Faithfulness** : Fidélité de la réponse aux documents
   - **Answer Relevancy** : Pertinence de la réponse
   - **Context Precision** : Précision du contexte récupéré
   - **Context Recall** : Rappel du contexte pertinent
4. Sauvegarder les résultats dans `ragas_evaluation_results.csv`

## Structure du projet

```
diabete-rag/
├── documents/           # Documents PDF à indexer
├── storage/            # Index vectoriel persistant (créé automatiquement)
├── app.py              # Application Streamlit (Frontend Web)
├── main.py             # Interface CLI
├── documents.py        # Classe pour créer l'index avec nettoyage
├── llm.py              # Gestionnaire du LLM
├── text_cleaner.py     # Module de nettoyage et prétraitement des PDFs
├── preprocess_pdfs.py  # Script d'analyse des documents
├── test_ragas.py       # Évaluation avec RAGAS (avec nettoyage intégré)
├── requirements.txt    # Dépendances Python
└── .env               # Variables d'environnement (clé API)
```

## Fonctionnalités de nettoyage des PDFs

Le système intègre désormais un nettoyage automatique des documents PDF pour améliorer la qualité des réponses :

- ✅ **Headers/footers répétitifs** - Détection et suppression automatique
- ✅ **Numéros de page** - Suppression des numéros de page parasites
- ✅ **Caractères spéciaux** - Normalisation des espaces insécables, apostrophes, etc.
- ✅ **Mots coupés** - Correction des mots séparés par des tirets en fin de ligne
- ✅ **URLs et emails** - Suppression optionnelle
- ✅ **Termes médicaux** - Normalisation optionnelle des abréviations

## 📚 Guide d'utilisation pas à pas

### Comment ajouter de nouveaux documents ?

1. **Placer les PDFs** dans le dossier `documents/`
2. **Supprimer l'ancien index** (optionnel, pour forcer la réindexation):
   ```bash
   # Windows
   rmdir /s storage

   # Linux/MacOS
   rm -rf storage
   ```
3. **Relancer l'application** - l'index sera recréé automatiquement

### Comment changer de modèle LLM ?

1. **Ouvrir le fichier** [.env](.env)
2. **Modifier** les variables selon le modèle souhaité:
   ```env
   LLM_MODEL_NAME=gpt-4  # ou llama-3.1-70b-versatile, etc.
   ```
3. **Redémarrer l'application**

### Comment obtenir une clé API ?

#### Pour Groq (gratuit):
1. Créer un compte sur [console.groq.com](https://console.groq.com)
2. Aller dans "API Keys"
3. Créer une nouvelle clé
4. Copier la clé dans votre fichier `.env`

#### Pour Azure OpenAI:
1. Créer une ressource Azure OpenAI dans le portail Azure
2. Déployer un modèle (GPT-4, etc.)
3. Récupérer la clé et l'endpoint dans les paramètres
4. Ajouter dans `.env`:
   ```env
   API_KEY=votre_clé_azure
   API_BASE_URL=https://votre-ressource.openai.azure.com/openai/v1/
   ```

### Comment améliorer la qualité des réponses ?

1. **Nettoyer les PDFs** avant indexation:
   ```bash
   python preprocess_pdfs.py --preview
   ```

2. **Activer les options de nettoyage** dans votre code:
   ```python
   embedder = EmbedderRag(
       clean_text=True,
       remove_urls=True,
       normalize_medical=True
   )
   ```

3. **Évaluer avec RAGAS** pour mesurer la performance:
   ```bash
   python test_ragas.py
   ```

4. **Ajuster la température** du LLM dans [llm.py](llm.py:31):
   - Température basse (0.1-0.5): Réponses plus précises et factuelles
   - Température haute (0.7-1.0): Réponses plus créatives

## ❓ Dépannage

### Erreur: "La clé API_KEY n'est pas définie"
- Vérifiez que le fichier `.env` existe à la racine du projet
- Assurez-vous que la variable `API_KEY` est bien définie dans `.env`
- Redémarrez l'application après modification du `.env`

### Erreur: "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Les réponses sont de mauvaise qualité
- Analysez vos documents avec `python preprocess_pdfs.py --preview`
- Activez le nettoyage automatique avec `clean_text=True`
- Essayez un modèle LLM plus performant (GPT-4 vs GPT-3.5)
- Réduisez la température du LLM pour plus de précision

### L'application Streamlit ne se lance pas
```bash
# Vérifier que Streamlit est installé
pip install streamlit

# Lancer avec mode debug
streamlit run app.py --logger.level=debug
```
