# Rapport de Dépannage - Évaluation RAG avec RAGAS

Ce rapport détaille les étapes de dépannage et les résolutions apportées aux problèmes rencontrés lors de l'exécution du script d'évaluation RAG (`test_ragas.py`) en utilisant `llama-index` et `ragas`.

---

### 1. Problème Initial : `ValueError: No API key found for OpenAI`

*   **Symptôme :** Le script s'arrêtait immédiatement au démarrage avec une erreur indiquant qu'aucune clé API OpenAI n'avait été trouvée.
*   **Analyse :** Votre configuration (`llm.py`) utilisait `OpenAILike` pour se connecter à un service Azure OpenAI, mais la bibliothèque `llama-index` tentait par défaut d'initialiser un modèle OpenAI standard lorsque le `QueryEngine` était créé sans spécifier explicitement le LLM.
*   **Résolution :** Nous avons modifié `test_ragas.py` pour passer explicitement votre instance de LLM configurée (`self.llm`) lors de la création du `QueryEngine`, garantissant ainsi que `llama-index` utilisait le bon modèle et non son comportement par défaut.
    *   **Modification :** Passage de `query_engine = index.as_query_engine()` à `query_engine = index.as_query_engine(llm=self.llm)`.

---

### 2. Deuxième Problème : `openai.OpenAIError` lors de l'Évaluation `ragas`

*   **Symptôme :** Après la résolution du premier problème, une nouvelle erreur `openai.OpenAIError` est apparue, mais les logs indiquaient qu'elle provenait cette fois de la bibliothèque `ragas` elle-même, spécifiquement d'un module interne `langchain_openai`.
*   **Analyse :** `ragas` utilise des modèles d'embedding pour certaines de ses métriques (par exemple, `AnswerRelevancy`). Même si votre modèle d'embedding Ollama était configuré pour `llama-index`, `ragas` ne le reconnaissait pas automatiquement et essayait de charger ses propres `OpenAIEmbeddings` par défaut, ce qui nécessitait une clé API OpenAI.
*   **Explication à l'Utilisateur :** Il a été clarifié que `ragas` dépendait en interne de `langchain` pour gérer les modèles d'embedding non-OpenAI, et qu'il serait nécessaire d'utiliser cette abstraction.
*   **Résolution :**
    1.  Installation du paquet `langchain-ollama` (après avoir corrigé un problème de `pip` manquant dans l'environnement virtuel).
    2.  Modification de `test_ragas.py` pour importer `OllamaEmbeddings` depuis `langchain_ollama` (afin d'utiliser la version la plus récente et non dépréciée).
    3.  Création d'une instance de `LangchainEmbeddingsWrapper` autour de l'embedding Ollama et passage de cet objet **uniquement** à la métrique `AnswerRelevancy`, car c'est la seule métrique qui requiert explicitement un modèle d'embedding selon la documentation de `ragas`. Les autres métriques calculent leur score avec le LLM.

---

### 3. Troisième Problème : Erreur d'Attribut/Clé lors de l'Affichage des Résultats

*   **Symptôme :** Une fois les problèmes de configuration des modèles résolus, l'évaluation `ragas` s'exécutait entièrement, mais le script échouait lors de la tentative d'affichage des scores moyens dans la fonction `display_results`. Des erreurs comme `AttributeError: 'EvaluationResult' object has no attribute 'items'` ou `KeyError: 0` sont apparues, suggérant une incompréhension de la structure de l'objet `result` retourné par `ragas.evaluate`.
*   **Analyse :** Après avoir ajouté des instructions de débogage, il a été déterminé que l'objet `result` était bien de type `ragas.dataset_schema.EvaluationResult`. Cet objet ne se comportait pas exactement comme un dictionnaire directement itérable pour obtenir les scores agrégés, ni son attribut `.scores` n'était dans le format attendu pour une itération directe.
*   **Résolution Finale :** La méthode `display_results` a été entièrement réécrite pour tirer parti de la capacité de l'objet `EvaluationResult` à être converti directement en un DataFrame pandas. Les scores moyens sont ensuite calculés en prenant la moyenne de chaque colonne numérique du DataFrame résultant.
    *   **Modification :** La fonction `display_results` a été mise à jour pour utiliser `df = result.to_pandas()` puis `df[metric].mean(skipna=True)` pour afficher les scores moyens.

---

### Conclusion

Toutes les erreurs ont été résolues. Votre script `test_ragas.py` s'exécute désormais avec succès, configure correctement votre modèle Azure/Ollama, effectue l'évaluation `ragas` et affiche les résultats de manière appropriée.

**Note sur l'optimisation des tokens :** Comme discuté précédemment, pour réduire la consommation de tokens avec des modèles locaux, vous pourriez envisager de limiter la concurrence des requêtes en utilisant `run_config=RunConfig(max_workers=1)` dans l'appel à `ragas.evaluate`.
