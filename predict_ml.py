"""
Machine Learning prediction script for disease prediction.
Uses trained TF-IDF + RandomForest model.
"""

import os
import numpy as np
import joblib
import pickle
from typing import Dict, List, Tuple, Optional
from preprocessing.preprocess import TextPreprocessor, load_preprocessor

class MLDiseasePredictor:
    """ML-based disease prediction using TF-IDF + Random Forest."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Load all required models and preprocessors."""
        try:
            # Load ML model
            model_path = os.path.join(self.models_dir, "ml_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ML model not found at {model_path}")
            self.model = joblib.load(model_path)
            
            # Load TF-IDF vectorizer
            vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
            self.vectorizer = load_preprocessor(vectorizer_path)
            
            # Load label encoder
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
            self.label_encoder = load_preprocessor(encoder_path)
            
            # Load feature names (optional)
            feature_names_path = os.path.join(self.models_dir, "feature_names.pkl")
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
            
            self.is_loaded = True
            print("ML models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading ML models: {e}")
            self.is_loaded = False
            return False
    
    def predict_single(self, symptom_text: str) -> Dict:
        """
        Predict disease for a single symptom description.
        
        Args:
            symptom_text (str): Raw symptom description
            
        Returns:
            Dict: Prediction results with disease, confidence, and top predictions
        """
        if not self.is_loaded:
            if not self.load_models():
                return {"error": "Models not loaded"}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(symptom_text)
            
            if not cleaned_text.strip():
                return {
                    "error": "No valid symptoms found after preprocessing",
                    "original_text": symptom_text,
                    "cleaned_text": cleaned_text
                }
            
            # Convert to TF-IDF features
            X = self.vectorizer.transform([cleaned_text])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            predicted_class = self.model.predict(X)[0]
            
            # Get disease name
            predicted_disease = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = probabilities[predicted_class]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_predictions = []
            
            for idx in top_indices:
                disease = self.label_encoder.inverse_transform([idx])[0]
                prob = probabilities[idx]
                top_predictions.append({
                    "disease": disease,
                    "confidence": float(prob)
                })
            
            return {
                "predicted_disease": predicted_disease,
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "original_text": symptom_text,
                "cleaned_text": cleaned_text,
                "model_type": "machine_learning"
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "original_text": symptom_text
            }
    
    def predict_batch(self, symptom_texts: List[str]) -> List[Dict]:
        """
        Predict diseases for multiple symptom descriptions.
        
        Args:
            symptom_texts (List[str]): List of symptom descriptions
            
        Returns:
            List[Dict]: List of prediction results
        """
        if not self.is_loaded:
            if not self.load_models():
                return [{"error": "Models not loaded"} for _ in symptom_texts]
        
        results = []
        for text in symptom_texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def get_feature_importance(self, symptom_text: str, top_n: int = 10) -> Dict:
        """
        Get feature importance for a specific prediction.
        
        Args:
            symptom_text (str): Symptom description
            top_n (int): Number of top features to return
            
        Returns:
            Dict: Feature importance analysis
        """
        if not self.is_loaded or self.feature_names is None:
            return {"error": "Models or feature names not loaded"}
        
        try:
            # Preprocess and vectorize
            cleaned_text = self.preprocessor.clean_text(symptom_text)
            X = self.vectorizer.transform([cleaned_text])
            
            # Get prediction
            prediction = self.predict_single(symptom_text)
            if "error" in prediction:
                return prediction
            
            # Get feature importance from the model
            feature_importance = self.model.feature_importances_
            
            # Get active features in this sample
            active_features = X.toarray()[0]
            feature_contributions = active_features * feature_importance
            
            # Get top contributing features
            top_indices = np.argsort(feature_contributions)[::-1][:top_n]
            top_features = []
            
            for idx in top_indices:
                if active_features[idx] > 0:  # Only include active features
                    top_features.append({
                        "feature": self.feature_names[idx],
                        "importance": float(feature_importance[idx]),
                        "contribution": float(feature_contributions[idx]),
                        "tfidf_score": float(active_features[idx])
                    })
            
            return {
                "prediction": prediction,
                "top_features": top_features,
                "total_active_features": int(np.sum(active_features > 0))
            }
            
        except Exception as e:
            return {"error": f"Feature analysis failed: {str(e)}"}
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"error": "Models not loaded"}
        
        return {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": len(self.feature_names) if self.feature_names else "Unknown",
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "vectorizer_type": "TfidfVectorizer",
            "max_features": self.vectorizer.max_features
        }

# Standalone prediction functions for easy import
_predictor_instance = None

def get_predictor() -> MLDiseasePredictor:
    """Get or create predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLDiseasePredictor()
    return _predictor_instance

def predict_with_ml(symptom_text: str) -> Dict:
    """
    Standalone function for ML prediction.
    
    Args:
        symptom_text (str): Symptom description
        
    Returns:
        Dict: Prediction result
    """
    predictor = get_predictor()
    return predictor.predict_single(symptom_text)

def predict_batch_ml(symptom_texts: List[str]) -> List[Dict]:
    """
    Standalone function for batch ML prediction.
    
    Args:
        symptom_texts (List[str]): List of symptom descriptions
        
    Returns:
        List[Dict]: List of prediction results
    """
    predictor = get_predictor()
    return predictor.predict_batch(symptom_texts)

def analyze_prediction_ml(symptom_text: str, top_features: int = 10) -> Dict:
    """
    Standalone function for prediction analysis.
    
    Args:
        symptom_text (str): Symptom description
        top_features (int): Number of top features to analyze
        
    Returns:
        Dict: Analysis result
    """
    predictor = get_predictor()
    return predictor.get_feature_importance(symptom_text, top_features)

# Example usage and testing
if __name__ == "__main__":
    print("Testing ML Disease Prediction...")
    print("=" * 50)
    
    # Test samples
    test_symptoms = [
        "I have high fever and body pain with headache",
        "Experiencing severe cough and cold symptoms",
        "Feeling nauseous with stomach ache and diarrhea",
        "Having chest pain and difficulty breathing",
        "Skin rash and itching all over body",
        "Severe headache with sensitivity to light"
    ]
    
    predictor = MLDiseasePredictor()
    
    if predictor.load_models():
        print("Models loaded successfully!")
        print(f"Model info: {predictor.get_model_info()}")
        print("\nTesting predictions:")
        print("-" * 50)
        
        for symptom in test_symptoms:
            result = predictor.predict_single(symptom)
            
            if "error" not in result:
                print(f"Symptom: {symptom}")
                print(f"Predicted: {result['predicted_disease']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Top 3 predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"  {i}. {pred['disease']} ({pred['confidence']:.3f})")
                print("-" * 50)
            else:
                print(f"Error for '{symptom}': {result['error']}")
        
        # Test feature analysis
        print("\nFeature Analysis Example:")
        analysis = predictor.get_feature_importance(test_symptoms[0])
        if "error" not in analysis:
            print(f"Top features for: {test_symptoms[0]}")
            for feature in analysis['top_features'][:5]:
                print(f"  {feature['feature']}: {feature['contribution']:.4f}")
    else:
        print("Failed to load models. Please train the models first.")
        print("Run: python train_ml_model.py")
