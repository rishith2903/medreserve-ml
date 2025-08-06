# 🤖 MedReserve ML Service - Deployment Guide

## 🚀 Quick Deploy to Render

### Prerequisites
- GitHub repository with ML service code
- Render.com account (free tier available)

### Step 1: Repository Setup

If deploying from a separate ML repository (`medreserve-ml`):

1. **Create/Update the ML Repository**
   ```bash
   # If creating a new repository
   git clone https://github.com/rishith2903/medreserve-ml.git
   cd medreserve-ml
   
   # Copy ML service files
   cp -r /path/to/MedReserve/backend/ml/* .
   
   # Commit and push
   git add .
   git commit -m "Add ML service with Docker deployment"
   git push origin main
   ```

### Step 2: Deploy to Render

1. **Connect Repository**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your `medreserve-ml` repository

2. **Configure Service**
   - **Name**: `medreserve-ml`
   - **Environment**: Docker
   - **Region**: Singapore (or your preferred region)
   - **Branch**: main
   - **Dockerfile Path**: `./Dockerfile`

3. **Environment Variables** (Auto-configured from render.yaml)
   - `PORT`: 5001
   - `DEBUG`: false
   - `ENVIRONMENT`: production
   - `CORS_ALLOWED_ORIGINS`: Your frontend URLs

### Step 3: Verify Deployment

1. **Check Health Endpoint**
   ```bash
   curl https://medreserve-ml.onrender.com/health
   ```

2. **Test ML Predictions**
   ```bash
   curl -X POST https://medreserve-ml.onrender.com/predict/specialization \
     -H "Content-Type: application/json" \
     -d '{"symptoms": "chest pain and shortness of breath"}'
   ```

## 🔧 Configuration Files

### Essential Files for Deployment:
- `Dockerfile` - Container build instructions
- `render.yaml` - Render service configuration
- `requirements-minimal.txt` - Essential Python dependencies
- `start.py` - Startup script with model training
- `.dockerignore` - Optimizes build context

### File Structure:
```
medreserve-ml/
├── Dockerfile
├── render.yaml
├── requirements-minimal.txt
├── start.py
├── .dockerignore
├── api/
│   └── ml_api.py
├── train/
├── predict/
├── nlp/
├── utils/
└── models/ (generated during deployment)
```

## 🚀 Deployment Process

1. **Build Stage**:
   - Installs Python dependencies
   - Sets up virtual environment
   - Optimizes for production

2. **Runtime Stage**:
   - Downloads NLTK data
   - Trains ML models (if not present)
   - Starts Flask API server

3. **Health Checks**:
   - Container health check every 30s
   - API health endpoint at `/health`

## 🔍 Troubleshooting

### Common Issues:

1. **Build Timeout**
   ```
   Solution: Use requirements-minimal.txt for faster builds
   ```

2. **Model Training Fails**
   ```
   Check logs: Service will start with fallback mode
   Models will be retrained on next restart
   ```

3. **Memory Issues**
   ```
   Upgrade to Render paid plan for more memory
   Or optimize model size in training scripts
   ```

4. **NLTK Data Missing**
   ```
   Startup script automatically downloads required data
   Check logs for download status
   ```

### Debugging Commands:

```bash
# Check service logs
curl https://medreserve-ml.onrender.com/health

# Test specific endpoints
curl -X POST https://medreserve-ml.onrender.com/predict/specialization \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "test symptoms"}'

# Check model status
curl https://medreserve-ml.onrender.com/models/info
```

## 📊 Performance Optimization

### For Production:
1. **Use minimal dependencies** (requirements-minimal.txt)
2. **Pre-trained models** (commit trained models to reduce startup time)
3. **Caching** (Enable model caching in production)
4. **Monitoring** (Set up health checks and alerts)

### Resource Usage:
- **Memory**: ~512MB (with models loaded)
- **CPU**: 1 vCPU (sufficient for free tier)
- **Storage**: ~100MB (models + dependencies)
- **Startup Time**: 2-5 minutes (including model training)

## 🔗 Integration

### Backend Integration:
Update your main backend's ML service URL:

```yaml
# In main backend render.yaml
envVars:
  - key: ML_SERVICE_URL
    value: https://medreserve-ml.onrender.com
```

### Frontend Integration:
The ML service provides REST APIs that can be called directly or through your main backend.

## 📞 Support

If deployment fails:
1. Check Render build logs
2. Verify all required files are present
3. Test locally with Docker first
4. Check GitHub repository permissions
