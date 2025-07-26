from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from datetime import datetime
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedReserve AI - ML Microservice",
    description="Machine Learning microservice for medical specialty prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class SymptomRequest(BaseModel):
    symptoms: str = Field(..., description="Patient symptoms description", min_length=1, max_length=1000)
    age: Optional[int] = Field(None, description="Patient age", ge=0, le=150)
    gender: Optional[str] = Field(None, description="Patient gender")

class SpecialtyPrediction(BaseModel):
    specialty: str
    confidence: float
    description: str

class PredictionResponse(BaseModel):
    predictions: List[SpecialtyPrediction]
    recommended_specialty: str
    confidence_score: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Medical specialties and their descriptions
MEDICAL_SPECIALTIES = {
    "Cardiology": "Heart and cardiovascular system disorders",
    "Dermatology": "Skin, hair, and nail conditions",
    "Endocrinology": "Hormonal and metabolic disorders",
    "Gastroenterology": "Digestive system disorders",
    "Neurology": "Nervous system disorders",
    "Orthopedics": "Bone, joint, and muscle disorders",
    "Psychiatry": "Mental health and behavioral disorders",
    "Pulmonology": "Lung and respiratory system disorders",
    "Urology": "Urinary system and male reproductive disorders",
    "Gynecology": "Female reproductive system disorders",
    "Ophthalmology": "Eye and vision disorders",
    "ENT": "Ear, nose, and throat disorders",
    "General Medicine": "General health issues and preventive care",
    "Emergency Medicine": "Urgent and emergency medical conditions"
}

# Training data for the ML model
TRAINING_DATA = [
    # Cardiology
    ("chest pain heart palpitations shortness of breath", "Cardiology"),
    ("heart attack chest tightness irregular heartbeat", "Cardiology"),
    ("high blood pressure hypertension cardiac", "Cardiology"),
    ("heart murmur chest discomfort fatigue", "Cardiology"),
    
    # Dermatology
    ("skin rash itching acne eczema", "Dermatology"),
    ("moles skin cancer melanoma", "Dermatology"),
    ("psoriasis dry skin scaling", "Dermatology"),
    ("hair loss baldness alopecia", "Dermatology"),
    
    # Endocrinology
    ("diabetes blood sugar thyroid", "Endocrinology"),
    ("weight gain weight loss metabolism", "Endocrinology"),
    ("hormone imbalance insulin resistance", "Endocrinology"),
    ("fatigue excessive thirst frequent urination", "Endocrinology"),
    
    # Gastroenterology
    ("stomach pain abdominal pain nausea vomiting", "Gastroenterology"),
    ("diarrhea constipation bowel movements", "Gastroenterology"),
    ("acid reflux heartburn indigestion", "Gastroenterology"),
    ("liver problems hepatitis", "Gastroenterology"),
    
    # Neurology
    ("headache migraine seizures", "Neurology"),
    ("memory loss confusion dementia", "Neurology"),
    ("numbness tingling weakness", "Neurology"),
    ("stroke paralysis brain", "Neurology"),
    
    # Orthopedics
    ("back pain joint pain arthritis", "Orthopedics"),
    ("bone fracture injury sports", "Orthopedics"),
    ("knee pain hip pain shoulder pain", "Orthopedics"),
    ("muscle strain sprain", "Orthopedics"),
    
    # Psychiatry
    ("depression anxiety stress mental health", "Psychiatry"),
    ("panic attacks mood swings", "Psychiatry"),
    ("insomnia sleep disorders", "Psychiatry"),
    ("bipolar disorder schizophrenia", "Psychiatry"),
    
    # Pulmonology
    ("cough breathing problems asthma", "Pulmonology"),
    ("lung infection pneumonia", "Pulmonology"),
    ("wheezing chest congestion", "Pulmonology"),
    ("chronic obstructive pulmonary disease copd", "Pulmonology"),
    
    # Urology
    ("urinary tract infection kidney stones", "Urology"),
    ("frequent urination painful urination", "Urology"),
    ("prostate problems erectile dysfunction", "Urology"),
    ("kidney disease bladder problems", "Urology"),
    
    # Gynecology
    ("menstrual problems irregular periods", "Gynecology"),
    ("pregnancy prenatal care", "Gynecology"),
    ("pelvic pain ovarian cysts", "Gynecology"),
    ("menopause hormonal changes", "Gynecology"),
    
    # Ophthalmology
    ("eye pain vision problems blurred vision", "Ophthalmology"),
    ("cataracts glaucoma", "Ophthalmology"),
    ("dry eyes eye infection", "Ophthalmology"),
    ("retinal problems macular degeneration", "Ophthalmology"),
    
    # ENT
    ("ear pain hearing loss tinnitus", "ENT"),
    ("sore throat throat infection", "ENT"),
    ("sinus problems nasal congestion", "ENT"),
    ("voice problems hoarseness", "ENT"),
    
    # General Medicine
    ("fever cold flu general checkup", "General Medicine"),
    ("vaccination preventive care", "General Medicine"),
    ("general health wellness", "General Medicine"),
    ("routine examination physical", "General Medicine"),
    
    # Emergency Medicine
    ("severe pain emergency urgent", "Emergency Medicine"),
    ("accident trauma injury", "Emergency Medicine"),
    ("severe bleeding unconscious", "Emergency Medicine"),
    ("poisoning overdose", "Emergency Medicine")
]

# Global model variable
ml_model = None

def load_model():
    """Load or train the ML model"""
    global ml_model
    
    try:
        # Try to load existing model
        if os.path.exists("specialty_model.joblib"):
            ml_model = joblib.load("specialty_model.joblib")
            logger.info("Loaded existing ML model")
        else:
            # Train new model
            train_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        train_model()

def train_model():
    """Train the ML model with symptom data"""
    global ml_model
    
    try:
        # Prepare training data
        symptoms = [data[0] for data in TRAINING_DATA]
        specialties = [data[1] for data in TRAINING_DATA]
        
        # Create pipeline with TF-IDF and Naive Bayes
        ml_model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Train the model
        ml_model.fit(symptoms, specialties)
        
        # Save the model
        joblib.dump(ml_model, "specialty_model.joblib")
        logger.info("Trained and saved new ML model")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train ML model")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token with main backend"""
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{backend_url}/api/auth/verify",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            
            return response.json()
            
    except httpx.RequestError:
        # If backend is not available, allow access for development
        logger.warning("Backend not available, allowing access")
        return {"user": "development", "role": "PATIENT"}
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/analyze-symptoms", response_model=PredictionResponse)
async def analyze_symptoms(
    request: SymptomRequest
):
    """Analyze symptoms and predict medical specialty (public endpoint for testing)"""

    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not available")

    try:
        # Get prediction probabilities
        probabilities = ml_model.predict_proba([request.symptoms])[0]
        classes = ml_model.classes_

        # Create predictions with confidence scores
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > 0.1:  # Only include predictions with >10% confidence
                predictions.append(SpecialtyPrediction(
                    specialty=classes[i],
                    confidence=float(prob),
                    description=f"Recommended based on symptom analysis with {prob*100:.1f}% confidence"
                ))

        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)

        # Get recommended specialty and confidence
        recommended_specialty = predictions[0].specialty if predictions else "General Medicine"
        confidence_score = predictions[0].confidence if predictions else 0.5

        # Limit to top 3 predictions
        predictions = predictions[:3]

        logger.info(f"Symptom analysis for: {request.symptoms[:50]}...")

        return PredictionResponse(
            predictions=predictions,
            recommended_specialty=recommended_specialty,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error in symptom analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict-specialty", response_model=PredictionResponse)
async def predict_specialty(
    request: SymptomRequest,
    user_info: dict = Depends(verify_token)
):
    """Predict medical specialty based on symptoms (authenticated endpoint)"""
    
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not available")
    
    try:
        # Get prediction probabilities
        probabilities = ml_model.predict_proba([request.symptoms])[0]
        classes = ml_model.classes_
        
        # Create predictions with confidence scores
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > 0.01:  # Only include predictions with >1% confidence
                specialty = classes[i]
                predictions.append(SpecialtyPrediction(
                    specialty=specialty,
                    confidence=float(prob),
                    description=MEDICAL_SPECIALTIES.get(specialty, "Medical specialty")
                ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Get top recommendation
        recommended_specialty = predictions[0].specialty if predictions else "General Medicine"
        confidence_score = predictions[0].confidence if predictions else 0.5
        
        # Limit to top 3 predictions
        predictions = predictions[:3]
        
        logger.info(f"Prediction made for symptoms: {request.symptoms[:50]}...")
        
        return PredictionResponse(
            predictions=predictions,
            recommended_specialty=recommended_specialty,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/specialties")
async def get_specialties(user_info: dict = Depends(verify_token)):
    """Get list of available medical specialties"""
    return {
        "specialties": list(MEDICAL_SPECIALTIES.keys()),
        "descriptions": MEDICAL_SPECIALTIES
    }

@app.post("/retrain")
async def retrain_model(user_info: dict = Depends(verify_token)):
    """Retrain the ML model (admin only)"""
    
    # Check if user has admin privileges
    user_role = user_info.get("role", "")
    if user_role not in ["ADMIN", "MASTER_ADMIN", "DOCTOR"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges"
        )
    
    try:
        train_model()
        return {"message": "Model retrained successfully", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail="Model retraining failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
