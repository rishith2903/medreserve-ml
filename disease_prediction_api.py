"""
Production Disease Prediction API Service.
Integrates with MedReserve backend for disease prediction using ML and DL models.
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict_ml import predict_with_ml, predict_batch_ml, analyze_prediction_ml
    from predict_dl import predict_with_dl, predict_batch_dl, analyze_prediction_dl
    from ensemble_predictor import predict_disease, compare_predictions, get_ensemble_predictor
except ImportError as e:
    logging.error(f"Failed to import prediction modules: {e}")
    # Create fallback functions
    def predict_with_ml(text): return {"error": "ML model not available"}
    def predict_with_dl(text): return {"error": "DL model not available"}
    def predict_disease(text, method="ensemble"): return {"error": "Models not available"}
    def compare_predictions(text): return {"error": "Models not available"}
    def predict_batch_ml(texts): return [{"error": "ML model not available"} for _ in texts]
    def predict_batch_dl(texts): return [{"error": "DL model not available"} for _ in texts]
    def analyze_prediction_ml(text, top_features=10): return {"error": "ML model not available"}
    def analyze_prediction_dl(text): return {"error": "DL model not available"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SymptomRequest(BaseModel):
    symptoms: str = Field(..., description="Patient symptoms description", min_length=1, max_length=2000)
    method: Optional[str] = Field("ensemble", description="Prediction method: ml, dl, or ensemble")
    age: Optional[int] = Field(None, description="Patient age", ge=0, le=150)
    gender: Optional[str] = Field(None, description="Patient gender")

class BatchSymptomRequest(BaseModel):
    symptoms_list: List[str] = Field(..., description="List of symptom descriptions")
    method: Optional[str] = Field("ensemble", description="Prediction method: ml, dl, or ensemble")

class AnalysisRequest(BaseModel):
    symptoms: str = Field(..., description="Patient symptoms description", min_length=1, max_length=2000)
    analysis_type: Optional[str] = Field("ml", description="Analysis type: ml or dl")
    top_features: Optional[int] = Field(10, description="Number of top features to return", ge=1, le=50)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: Dict[str, bool]

# Initialize FastAPI app
app = FastAPI(
    title="MedReserve Disease Prediction API",
    description="AI-powered disease prediction from symptom descriptions using ML and DL models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Optional JWT token validation.
    In production, implement proper JWT validation here.
    """
    if credentials:
        # TODO: Implement JWT validation
        return {"token": credentials.credentials}
    return None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MedReserve Disease Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "POST /predict - Ensemble disease prediction",
            "predict_ml": "POST /predict/ml - ML-only prediction",
            "predict_dl": "POST /predict/dl - DL-only prediction",
            "predict_batch": "POST /predict/batch - Batch predictions",
            "compare": "POST /compare - Compare ML vs DL predictions",
            "analyze_ml": "POST /analyze/ml - ML feature analysis",
            "analyze_dl": "POST /analyze/dl - DL attention analysis",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test model availability
        test_result = predict_disease("test symptoms", method="ensemble")
        models_loaded = {
            "ensemble": "error" not in test_result,
            "ml": "error" not in predict_with_ml("test"),
            "dl": "error" not in predict_with_dl("test")
        }
        
        status = "healthy" if any(models_loaded.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded=models_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded={"ensemble": False, "ml": False, "dl": False}
        )

@app.post("/predict")
async def predict_disease_endpoint(
    request: SymptomRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Predict disease from symptom description using ensemble method.
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        logger.info(f"Disease prediction request: method={request.method}, symptoms_length={len(request.symptoms)}")
        
        result = predict_disease(request.symptoms, method=request.method)
        
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/ml")
async def predict_ml_endpoint(
    request: SymptomRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Predict using Machine Learning model only."""
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = predict_with_ml(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/predict/dl")
async def predict_dl_endpoint(
    request: SymptomRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Predict using Deep Learning model only."""
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = predict_with_dl(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DL prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_endpoint(
    request: BatchSymptomRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Predict diseases for multiple symptom descriptions.
    """
    try:
        if not request.symptoms_list:
            raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")
        
        if len(request.symptoms_list) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
        
        logger.info(f"Batch prediction request: method={request.method}, count={len(request.symptoms_list)}")
        
        if request.method == "ml":
            results = predict_batch_ml(request.symptoms_list)
        elif request.method == "dl":
            results = predict_batch_dl(request.symptoms_list)
        else:  # ensemble
            predictor = get_ensemble_predictor()
            results = predictor.predict_batch(request.symptoms_list, method="ensemble")
        
        return {
            "predictions": results,
            "total_processed": len(results),
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/compare")
async def compare_models_endpoint(
    request: SymptomRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Compare ML and DL model predictions side by side.
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = compare_predictions(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.post("/analyze/ml")
async def analyze_ml_endpoint(
    request: AnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Analyze ML prediction with feature importance.
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = analyze_prediction_ml(request.symptoms, top_features=request.top_features)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

@app.post("/analyze/dl")
async def analyze_dl_endpoint(
    request: AnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Analyze DL prediction with word importance.
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = analyze_prediction_dl(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DL analysis failed: {str(e)}")

@app.get("/models/info")
async def get_models_info(current_user: Optional[Dict] = Depends(get_current_user)):
    """Get information about loaded models."""
    try:
        predictor = get_ensemble_predictor()
        predictor.load_models()
        
        info = predictor.get_model_info()
        info["timestamp"] = datetime.now().isoformat()
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Disease Prediction API...")
    try:
        # Try to load models
        predictor = get_ensemble_predictor()
        load_status = predictor.load_models()
        logger.info(f"Model loading status: {load_status}")
    except Exception as e:
        logger.warning(f"Failed to load models on startup: {e}")

if __name__ == "__main__":
    print("Starting MedReserve Disease Prediction API...")
    print("API Documentation: http://localhost:8003/docs")
    print("Alternative docs: http://localhost:8003/redoc")
    
    uvicorn.run(
        "disease_prediction_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
