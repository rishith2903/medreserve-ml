# 🧠 Disease Prediction Pipeline Integration

## Overview

The Disease Prediction Pipeline is a comprehensive AI-powered system that predicts diseases from symptom descriptions using both Machine Learning (ML) and Deep Learning (DL) models. It's fully integrated with the MedReserve backend and frontend.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Spring Boot     │    │  FastAPI        │
│   (Port 3000)   │◄──►│  Backend         │◄──►│  Disease API    │
│                 │    │  (Port 8080)     │    │  (Port 8003)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │  ML/DL Models   │
                                │               │  • TF-IDF +     │
                                │               │    RandomForest │
                                │               │  • LSTM Neural  │
                                │               │    Network      │
                                │               └─────────────────┘
                                ▼
                       ┌─────────────────┐
                       │  PostgreSQL DB  │
                       │  (Port 5432)    │
                       └─────────────────┘
```

## Features

### 🤖 AI Models
- **Machine Learning**: TF-IDF vectorization + RandomForest classifier
- **Deep Learning**: Bidirectional LSTM with attention mechanism
- **Ensemble Method**: Weighted combination of ML and DL predictions

### 🔍 Prediction Methods
- **Single Prediction**: Get disease prediction from symptom description
- **Model Comparison**: Compare ML vs DL predictions side by side
- **Feature Analysis**: Understand which features/words influenced the prediction
- **Batch Processing**: Process multiple symptom descriptions at once

### 🛡️ Fallback System
- Keyword-based predictions when AI services are unavailable
- Graceful degradation with informative error messages
- Health monitoring and status reporting

## API Endpoints

### Spring Boot Backend (`/disease-prediction`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Ensemble disease prediction |
| POST | `/predict/ml` | ML-only prediction |
| POST | `/predict/dl` | DL-only prediction |
| POST | `/compare` | Compare ML vs DL models |
| POST | `/analyze` | Feature/word importance analysis |
| GET | `/health` | Service health check |

### FastAPI Service (`http://localhost:8003`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check with model status |
| POST | `/predict` | Ensemble prediction |
| POST | `/predict/ml` | ML prediction |
| POST | `/predict/dl` | DL prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/compare` | Model comparison |
| POST | `/analyze/ml` | ML feature analysis |
| POST | `/analyze/dl` | DL attention analysis |

## Request/Response Examples

### Prediction Request
```json
{
  "symptoms": "I have high fever, body pain, and severe headache",
  "method": "ensemble",
  "age": 30,
  "gender": "male"
}
```

### Prediction Response
```json
{
  "predictedDisease": "Viral Fever",
  "confidence": 0.85,
  "modelType": "ensemble",
  "topPredictions": [
    {
      "disease": "Viral Fever",
      "confidence": 0.85,
      "description": "Common viral infection with fever and body aches"
    },
    {
      "disease": "Influenza",
      "confidence": 0.72,
      "description": "Seasonal flu with respiratory symptoms"
    }
  ],
  "ensembleMethod": "weighted_average",
  "mlWeight": 0.6,
  "dlWeight": 0.4,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Frontend Integration

### SymptomChecker Component
Located at `frontend/src/pages/AI/SymptomChecker.jsx`

**Features:**
- Interactive symptom input with sample suggestions
- Real-time prediction with confidence visualization
- Model comparison interface
- Feature importance analysis
- Service health monitoring

**Usage:**
1. Navigate to `/symptom-checker` in the application
2. Enter symptom description
3. Choose prediction method (Ensemble, ML, or DL)
4. Click "Predict", "Compare", or "Analyze"
5. View results with confidence scores and explanations

## Setup and Installation

### 1. Install Dependencies
```bash
cd backend/ml
pip install -r requirements.txt
pip install fastapi uvicorn pydantic
```

### 2. Start Disease Prediction Service
```bash
cd backend/ml
python disease_prediction_api.py
```

### 3. Verify Integration
- Backend: `http://localhost:8080/disease-prediction/health`
- FastAPI: `http://localhost:8003/health`
- Frontend: Navigate to Symptom Checker page

## Configuration

### Environment Variables

**Backend (application.yml):**
```yaml
disease:
  prediction:
    service:
      url: ${DISEASE_PREDICTION_SERVICE_URL:http://localhost:8003}
    fallback:
      enabled: ${DISEASE_PREDICTION_FALLBACK_ENABLED:true}
```

**Frontend (.env files):**
```env
VITE_DISEASE_PREDICTION_SERVICE_URL=http://localhost:8003
```

## Model Training

### Training Data Format
```csv
symptoms,disease
"fever headache body pain",Viral Fever
"cough cold runny nose",Common Cold
"chest pain breathing difficulty",Respiratory Issue
```

### Training Commands
```bash
# Train ML model
python train_ml.py

# Train DL model  
python train_dl.py

# Train ensemble
python ensemble_predictor.py
```

## Monitoring and Health Checks

### Service Health
- **Healthy**: All models loaded and functional
- **Degraded**: Some models unavailable, fallback active
- **Unhealthy**: Service completely unavailable

### Fallback Behavior
When AI services are unavailable:
1. Keyword-based disease prediction
2. Confidence scores based on keyword matching
3. Informative error messages
4. Graceful degradation without breaking user experience

## Security Considerations

- JWT token validation (optional)
- Input sanitization and validation
- Rate limiting (configurable)
- CORS configuration for cross-origin requests

## Performance Optimization

- Model caching and lazy loading
- Batch processing for multiple predictions
- Asynchronous processing with FastAPI
- Connection pooling and timeout handling

## Troubleshooting

### Common Issues

1. **Service Unavailable**
   - Check if FastAPI service is running on port 8003
   - Verify network connectivity
   - Check logs for model loading errors

2. **Low Prediction Accuracy**
   - Retrain models with more diverse data
   - Adjust ensemble weights
   - Improve text preprocessing

3. **Slow Response Times**
   - Enable model caching
   - Use batch processing for multiple requests
   - Optimize model inference

### Logs and Debugging
- FastAPI logs: Check console output when running `disease_prediction_api.py`
- Spring Boot logs: Check application logs for integration errors
- Frontend logs: Check browser console for API call issues

## Future Enhancements

- [ ] Real-time model retraining
- [ ] Multi-language support
- [ ] Integration with medical databases
- [ ] Advanced ensemble methods
- [ ] Explainable AI features
- [ ] Mobile app integration
