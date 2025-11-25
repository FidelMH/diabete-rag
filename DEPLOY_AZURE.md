# Guide de Déploiement Azure

## Fichiers créés pour le déploiement

- ✅ `startup.txt` - Commande de démarrage Streamlit
- ✅ `.streamlit/config.toml` - Configuration Streamlit pour Azure
- ✅ `.deployment` - Configuration du build Oryx
- ✅ `requirements.txt` - Dépendances avec versions fixes

## Configuration Azure App Service requise

### 1. Variables d'environnement (Application Settings)

Dans Azure Portal > App Service > Configuration > Application settings, ajoutez :

```
API_KEY=<votre_clé_API>
API_BASE_URL=<votre_URL_API>
EMBEDDING_MODEL_NAME=text-embedding-3-large
LLM_MODEL_NAME=grok-4-fast-non-reasoning
```

### 2. Commande de démarrage (Startup Command)

Dans Azure Portal > App Service > Configuration > General settings :

```bash
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### 3. Paramètres recommandés

- **Plan tarifaire** : B2 ou supérieur (B1 peut être trop limité)
- **Python version** : 3.13
- **Stack** : Python

### 4. Configuration avancée (optionnel)

Si le build échoue à cause de timeouts, ajoutez ces settings :

```
SCM_COMMAND_IDLE_TIMEOUT=3600
POST_BUILD_SCRIPT_PATH=
WEBSITE_TIME_ZONE=Romance Standard Time
```

## Problèmes courants et solutions

### Build timeout
**Symptôme** : "Error: Failed to deploy web package"

**Solution** :
1. Augmentez le plan tarifaire (B2 minimum)
2. Ajoutez `SCM_COMMAND_IDLE_TIMEOUT=3600` dans les settings

### Out of Memory
**Symptôme** : Build échoue sans message clair

**Solution** :
1. Passez à un plan avec plus de RAM (B2: 3.5GB, B3: 7GB)
2. Réduisez les dépendances si possible

### App ne démarre pas
**Symptôme** : "Application Error"

**Solution** :
1. Vérifiez que la commande de startup est configurée
2. Vérifiez les logs : App Service > Monitoring > Log stream
3. Vérifiez que toutes les variables d'environnement sont définies

### Documents non trouvés
**Symptôme** : Erreur au chargement de l'index

**Solution** :
- Le dossier `documents/` doit être inclus dans le déploiement
- Le dossier `storage/` sera créé automatiquement par l'app

## Commandes de déploiement

### Via GitHub Actions (automatique)
```bash
git add .
git commit -m "Configure Azure deployment"
git push origin main
```

### Via Azure CLI (manuel)
```bash
az webapp up --name diabete-rag --resource-group <votre-rg> --runtime "PYTHON:3.13"
```

## Vérification du déploiement

1. Attendez la fin du build (peut prendre 10-15 minutes)
2. Accédez à l'URL : https://diabete-rag-btdegsg6cudre6g8.francecentral-01.azurewebsites.net
3. Vérifiez les logs si erreur : Portal > App Service > Log stream

## Optimisations post-déploiement

1. **Always On** : Activez pour éviter les cold starts
2. **Health check** : Configurez `/` comme endpoint de santé
3. **Scaling** : Activez l'auto-scaling si nécessaire
