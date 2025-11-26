# Azure Deployment Guide

## Deployment Files

- ✅ `startup.txt` - Streamlit startup command
- ✅ `.streamlit/config.toml` - Streamlit configuration for Azure
- ✅ `.deployment` - Oryx build configuration
- ✅ `requirements.txt` - Dependencies with fixed versions

## Required Azure App Service Configuration

### 1. Environment Variables (Application Settings)

In Azure Portal > App Service > Configuration > Application settings, add:

```
# Embedding Configuration (Azure OpenAI only for production)
EMBEDDING_PROVIDER=azure
EMBEDDING_API_KEY=<your_azure_key>
EMBEDDING_API_BASE=<your_azure_endpoint>
EMBEDDING_MODEL_NAME=text-embedding-3-large

# LLM Configuration
LLM_API_KEY=<your_azure_key>
LLM_API_BASE=<your_azure_endpoint>
LLM_MODEL_NAME=gpt-4
```

**⚠️ Important Notes:**
- **Always use `EMBEDDING_PROVIDER=azure` for production deployments**
- Local embeddings are NOT recommended in Azure due to:
  - Larger container size (includes PyTorch, models)
  - Slower cold starts
  - Higher memory usage
  - Model download on first run
- Use the same Azure OpenAI resource for both embeddings and LLM to simplify configuration

### 2. Startup Command

In Azure Portal > App Service > Configuration > General settings:

```bash
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### 3. Recommended Settings

- **Pricing tier**: B2 or higher (B1 may be too limited)
- **Python version**: 3.13
- **Stack**: Python
- **Always On**: Enable to avoid cold starts

### 4. Advanced Configuration (optional)

If build fails due to timeouts, add these settings:

```
SCM_COMMAND_IDLE_TIMEOUT=3600
POST_BUILD_SCRIPT_PATH=
WEBSITE_TIME_ZONE=Romance Standard Time
```

## Common Issues and Solutions

### Build Timeout
**Symptom**: "Error: Failed to deploy web package"

**Solution**:
1. Increase pricing tier (B2 minimum)
2. Add `SCM_COMMAND_IDLE_TIMEOUT=3600` in settings

### Out of Memory
**Symptom**: Build fails without clear message

**Solution**:
1. Upgrade to a plan with more RAM (B2: 3.5GB, B3: 7GB)
2. Reduce dependencies if possible

### App Won't Start
**Symptom**: "Application Error"

**Solution**:
1. Verify startup command is configured
2. Check logs: App Service > Monitoring > Log stream
3. Verify all environment variables are defined
4. Ensure `EMBEDDING_PROVIDER=azure` (local not supported)

### Documents Not Found
**Symptom**: Error when loading index

**Solution**:
- The `documents/` folder must be included in deployment
- The `storage/` folder will be created automatically by the app

### Missing Environment Variables
**Symptom**: "EMBEDDING_API_KEY not found" or "LLM_API_KEY not found"

**Solution**:
- Verify all environment variables are set in Application Settings
- Use the new variable names (not the old `API_KEY`, `API_BASE_URL`)
- Restart the App Service after adding variables

## Deployment Commands

### Via GitHub Actions (automatic)
```bash
git add .
git commit -m "Configure Azure deployment"
git push origin main
```

### Via Azure CLI (manual)
```bash
az webapp up --name diabete-rag --resource-group <your-rg> --runtime "PYTHON:3.13"
```

## Deployment Verification

1. Wait for build to complete (may take 10-15 minutes)
2. Access the URL: https://your-app-name.azurewebsites.net
3. Check logs if errors occur: Portal > App Service > Log stream
4. Verify the system loads and can query documents

## Post-Deployment Optimizations

1. **Always On**: Enable to avoid cold starts (Recommended for production)
2. **Health check**: Configure `/` as health check endpoint
3. **Scaling**: Enable auto-scaling if needed
4. **Monitoring**: Set up Application Insights for performance monitoring

## Environment Variables - Complete Reference

For quick copy-paste, here's the complete set of environment variables needed:

```bash
# Embedding Configuration
EMBEDDING_PROVIDER=azure
EMBEDDING_API_KEY=your_azure_api_key_here
EMBEDDING_API_BASE=https://your-resource.openai.azure.com/openai/v1/
EMBEDDING_MODEL_NAME=text-embedding-3-large

# LLM Configuration
LLM_API_KEY=your_azure_api_key_here
LLM_API_BASE=https://your-resource.openai.azure.com/openai/v1/
LLM_MODEL_NAME=gpt-4
```

**Note**: Replace `your-resource` with your actual Azure OpenAI resource name.

## Cost Optimization

- Use `text-embedding-3-small` instead of `text-embedding-3-large` for embeddings if cost is a concern
- Use `gpt-3.5-turbo` instead of `gpt-4` for the LLM (faster and cheaper, but lower quality)
- Monitor API usage in Azure Portal to track costs
