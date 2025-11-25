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

**Optionnel :** Analyser les documents avant indexation
```bash
python preprocess_pdfs.py --preview
```
Cette commande affiche des statistiques sur vos documents, détecte les problèmes potentiels et recommande les meilleures options de nettoyage.

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

### Problèmes résolus :
- ✅ **Erreurs headlines dans RAGAS** - Les métadonnées manquantes sont ajoutées automatiquement
- ✅ **Headers/footers répétitifs** - Détection et suppression automatique
- ✅ **Numéros de page** - Suppression des numéros de page parasites
- ✅ **Caractères spéciaux** - Normalisation des espaces insécables, apostrophes, etc.
- ✅ **Mots coupés** - Correction des mots séparés par des tirets en fin de ligne
- ✅ **URLs et emails** - Suppression optionnelle
- ✅ **Termes médicaux** - Normalisation optionnelle des abréviations

### Amélioration de la qualité :
Le nettoyage améliore la qualité des embeddings et des réponses du système RAG en :
- Réduisant le bruit dans les documents (10-20% de réduction typique)
- Améliorant la cohérence du texte
- Facilitant la recherche sémantique
- Évitant les erreurs lors de l'évaluation RAGAS