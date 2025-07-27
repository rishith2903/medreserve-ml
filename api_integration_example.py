"""
FastAPI integration example for Disease Prediction Pipeline.
Demonstrates how to integrate the prediction models with FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from predict_ml import predict_with_ml, predict_batch_ml, analyze_prediction_ml
from predict_dl import predict_with_dl, predict_batch_dl, analyze_prediction_dl
from ensemble_predictor import predict_disease, compare_predictions

# Pydantic models for request/response
class SymptomRequest(BaseModel):
    symptoms: str
    method: Optional[str] = "ensemble"  # "ml", "dl", or "ensemble"

class BatchSymptomRequest(BaseModel):
    symptoms_list: List[str]
    method: Optional[str] = "ensemble"

class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    model_type: str
    original_text: str

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_processed: int

class ComparisonResponse(BaseModel):
    symptom_text: str
    ml_prediction: Optional[Dict[str, Any]]
    dl_prediction: Optional[Dict[str, Any]]
    agreement: Optional[bool]
    confidence_difference: Optional[float]

# Initialize FastAPI app
app = FastAPI(
    title="Disease Prediction API",
    description="AI-powered disease prediction from symptom descriptions using ML and DL models",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Single prediction",
            "predict_batch": "POST /predict/batch - Batch predictions",
            "compare": "POST /compare - Compare ML vs DL predictions",
            "analyze_ml": "POST /analyze/ml - ML feature analysis",
            "analyze_dl": "POST /analyze/dl - DL attention analysis",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test a simple prediction to ensure models are working
        test_result = predict_disease("test symptoms", method="ensemble")
        
        return {
            "status": "healthy",
            "models_loaded": "error" not in test_result,
            "timestamp": "2024-01-01T00:00:00Z"  # In real app, use actual timestamp
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@app.post("/predict", response_model=Dict[str, Any])
async def predict_disease_endpoint(request: SymptomRequest):
    """
    Predict disease from symptom description.
    
    - **symptoms**: Text description of symptoms
    - **method**: Prediction method ("ml", "dl", or "ensemble")
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = predict_disease(request.symptoms, method=request.method)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/ml")
async def predict_ml_endpoint(request: SymptomRequest):
    """Predict using Machine Learning model only."""
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = predict_with_ml(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/predict/dl")
async def predict_dl_endpoint(request: SymptomRequest):
    """Predict using Deep Learning model only."""
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = predict_with_dl(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DL prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_endpoint(request: BatchSymptomRequest):
    """
    Predict diseases for multiple symptom descriptions.
    
    - **symptoms_list**: List of symptom descriptions
    - **method**: Prediction method ("ml", "dl", or "ensemble")
    """
    try:
        if not request.symptoms_list:
            raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")
        
        if len(request.symptoms_list) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
        
        if request.method == "ml":
            results = predict_batch_ml(request.symptoms_list)
        elif request.method == "dl":
            results = predict_batch_dl(request.symptoms_list)
        else:  # ensemble
            from ensemble_predictor import get_ensemble_predictor
            predictor = get_ensemble_predictor()
            results = predictor.predict_batch(request.symptoms_list, method="ensemble")
        
        return {
            "predictions": results,
            "total_processed": len(results),
            "method": request.method
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/compare")
async def compare_models_endpoint(request: SymptomRequest):
    """
    Compare ML and DL model predictions side by side.
    
    - **symptoms**: Text description of symptoms
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = compare_predictions(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.post("/analyze/ml")
async def analyze_ml_endpoint(request: SymptomRequest):
    """
    Analyze ML prediction with feature importance.
    
    - **symptoms**: Text description of symptoms
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = analyze_prediction_ml(request.symptoms, top_features=10)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

@app.post("/analyze/dl")
async def analyze_dl_endpoint(request: SymptomRequest):
    """
    Analyze DL prediction with word importance.
    
    - **symptoms**: Text description of symptoms
    """
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
        
        result = analyze_prediction_dl(request.symptoms)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DL analysis failed: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models."""
    try:
        from ensemble_predictor import get_ensemble_predictor
        predictor = get_ensemble_predictor()
        predictor.load_models()
        
        return predictor.get_model_info()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    print("Starting Disease Prediction API...")
    print("API Documentation will be available at: http://localhost:8003/docs")
    print("Alternative docs at: http://localhost:8003/redoc")

    # Run the API server
    uvicorn.run(
        "api_integration_example:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
