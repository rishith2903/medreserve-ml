"""
Ensemble Disease Prediction combining ML and DL models.
Provides unified interface for both models with ensemble predictions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from predict_ml import MLDiseasePredictor, predict_with_ml
from predict_dl import DLDiseasePredictor, predict_with_dl

class EnsembleDiseasePredictor:
    """Ensemble predictor combining ML and DL models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.ml_predictor = MLDiseasePredictor(models_dir)
        self.dl_predictor = DLDiseasePredictor(models_dir)
        self.is_loaded = False
        
    def load_models(self) -> Dict[str, bool]:
        """Load both ML and DL models."""
        ml_loaded = self.ml_predictor.load_models()
        dl_loaded = self.dl_predictor.load_models()
        
        self.is_loaded = ml_loaded or dl_loaded
        
        return {
            "ml_loaded": ml_loaded,
            "dl_loaded": dl_loaded,
            "ensemble_ready": self.is_loaded
        }
    
    def predict_single(self, symptom_text: str, method: str = "ensemble") -> Dict:
        """
        Predict disease using specified method.
        
        Args:
            symptom_text (str): Symptom description
            method (str): "ml", "dl", or "ensemble"
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_loaded:
            load_status = self.load_models()
            if not load_status["ensemble_ready"]:
                return {"error": "No models could be loaded"}
        
        if method == "ml":
            return self.ml_predictor.predict_single(symptom_text)
        elif method == "dl":
            return self.dl_predictor.predict_single(symptom_text)
        elif method == "ensemble":
            return self._ensemble_predict(symptom_text)
        else:
            return {"error": f"Unknown method: {method}. Use 'ml', 'dl', or 'ensemble'"}
    
    def _ensemble_predict(self, symptom_text: str) -> Dict:
        """Combine predictions from both models."""
        results = {
            "original_text": symptom_text,
            "ensemble_method": "weighted_average",
            "individual_predictions": {}
        }
        
        # Get ML prediction
        ml_result = None
        if self.ml_predictor.is_loaded:
            ml_result = self.ml_predictor.predict_single(symptom_text)
            results["individual_predictions"]["ml"] = ml_result
        
        # Get DL prediction
        dl_result = None
        if self.dl_predictor.is_loaded:
            dl_result = self.dl_predictor.predict_single(symptom_text)
            results["individual_predictions"]["dl"] = dl_result
        
        # Handle cases where only one model is available
        if ml_result and "error" not in ml_result and (not dl_result or "error" in dl_result):
            results.update(ml_result)
            results["ensemble_method"] = "ml_only"
            return results
        
        if dl_result and "error" not in dl_result and (not ml_result or "error" in ml_result):
            results.update(dl_result)
            results["ensemble_method"] = "dl_only"
            return results
        
        # Both models failed
        if (not ml_result or "error" in ml_result) and (not dl_result or "error" in dl_result):
            return {
                "error": "Both models failed to make predictions",
                "ml_error": ml_result.get("error") if ml_result else "Model not loaded",
                "dl_error": dl_result.get("error") if dl_result else "Model not loaded"
            }
        
        # Ensemble prediction - combine both models
        ensemble_prediction = self._combine_predictions(ml_result, dl_result)
        results.update(ensemble_prediction)
        
        return results
    
    def _combine_predictions(self, ml_result: Dict, dl_result: Dict) -> Dict:
        """Combine ML and DL predictions using weighted averaging."""
        # Extract disease predictions and confidences
        ml_diseases = {pred["disease"]: pred["confidence"] for pred in ml_result["top_predictions"]}
        dl_diseases = {pred["disease"]: pred["confidence"] for pred in dl_result["top_predictions"]}
        
        # Combine predictions with weights (can be tuned based on model performance)
        ml_weight = 0.6  # ML models often more interpretable
        dl_weight = 0.4  # DL models capture complex patterns
        
        combined_scores = {}
        all_diseases = set(ml_diseases.keys()) | set(dl_diseases.keys())
        
        for disease in all_diseases:
            ml_score = ml_diseases.get(disease, 0.0)
            dl_score = dl_diseases.get(disease, 0.0)
            
            # Weighted average
            combined_score = (ml_score * ml_weight + dl_score * dl_weight)
            combined_scores[disease] = combined_score
        
        # Sort by combined score
        sorted_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        top_predictions = []
        for disease, score in sorted_predictions[:5]:  # Top 5
            top_predictions.append({
                "disease": disease,
                "confidence": float(score),
                "ml_confidence": float(ml_diseases.get(disease, 0.0)),
                "dl_confidence": float(dl_diseases.get(disease, 0.0))
            })
        
        return {
            "predicted_disease": sorted_predictions[0][0],
            "confidence": float(sorted_predictions[0][1]),
            "top_predictions": top_predictions,
            "model_type": "ensemble",
            "ml_weight": ml_weight,
            "dl_weight": dl_weight
        }
    
    def predict_batch(self, symptom_texts: List[str], method: str = "ensemble") -> List[Dict]:
        """Predict diseases for multiple symptom descriptions."""
        return [self.predict_single(text, method) for text in symptom_texts]
    
    def compare_models(self, symptom_text: str) -> Dict:
        """Compare predictions from both models side by side."""
        if not self.is_loaded:
            load_status = self.load_models()
            if not load_status["ensemble_ready"]:
                return {"error": "No models could be loaded"}
        
        comparison = {
            "symptom_text": symptom_text,
            "ml_prediction": None,
            "dl_prediction": None,
            "agreement": None,
            "confidence_difference": None
        }
        
        # Get predictions
        if self.ml_predictor.is_loaded:
            comparison["ml_prediction"] = self.ml_predictor.predict_single(symptom_text)
        
        if self.dl_predictor.is_loaded:
            comparison["dl_prediction"] = self.dl_predictor.predict_single(symptom_text)
        
        # Analyze agreement
        if (comparison["ml_prediction"] and "error" not in comparison["ml_prediction"] and
            comparison["dl_prediction"] and "error" not in comparison["dl_prediction"]):
            
            ml_disease = comparison["ml_prediction"]["predicted_disease"]
            dl_disease = comparison["dl_prediction"]["predicted_disease"]
            ml_conf = comparison["ml_prediction"]["confidence"]
            dl_conf = comparison["dl_prediction"]["confidence"]
            
            comparison["agreement"] = ml_disease == dl_disease
            comparison["confidence_difference"] = abs(ml_conf - dl_conf)
            
            if comparison["agreement"]:
                comparison["consensus"] = {
                    "disease": ml_disease,
                    "avg_confidence": (ml_conf + dl_conf) / 2
                }
        
        return comparison
    
    def get_model_info(self) -> Dict:
        """Get information about both models."""
        info = {
            "ensemble_info": {
                "ml_weight": 0.6,
                "dl_weight": 0.4,
                "combination_method": "weighted_average"
            }
        }
        
        if self.ml_predictor.is_loaded:
            info["ml_model"] = self.ml_predictor.get_model_info()
        
        if self.dl_predictor.is_loaded:
            info["dl_model"] = self.dl_predictor.get_model_info()
        
        return info

# Standalone functions for easy import
_ensemble_instance = None

def get_ensemble_predictor() -> EnsembleDiseasePredictor:
    """Get or create ensemble predictor instance."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = EnsembleDiseasePredictor()
    return _ensemble_instance

def predict_disease(symptom_text: str, method: str = "ensemble") -> Dict:
    """
    Standalone function for disease prediction.
    
    Args:
        symptom_text (str): Symptom description
        method (str): "ml", "dl", or "ensemble"
        
    Returns:
        Dict: Prediction result
    """
    predictor = get_ensemble_predictor()
    return predictor.predict_single(symptom_text, method)

def compare_predictions(symptom_text: str) -> Dict:
    """
    Compare ML and DL predictions for a symptom description.
    
    Args:
        symptom_text (str): Symptom description
        
    Returns:
        Dict: Comparison results
    """
    predictor = get_ensemble_predictor()
    return predictor.compare_models(symptom_text)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Ensemble Disease Prediction...")
    print("=" * 60)
    
    # Test samples
    test_symptoms = [
        "I have high fever and body pain with headache",
        "Experiencing severe cough and cold symptoms",
        "Feeling nauseous with stomach ache and diarrhea",
        "Having chest pain and difficulty breathing"
    ]
    
    ensemble = EnsembleDiseasePredictor()
    load_status = ensemble.load_models()
    
    print(f"Model loading status: {load_status}")
    
    if load_status["ensemble_ready"]:
        print("\nTesting ensemble predictions:")
        print("-" * 50)
        
        for symptom in test_symptoms:
            print(f"\nSymptom: {symptom}")
            
            # Ensemble prediction
            ensemble_result = ensemble.predict_single(symptom, "ensemble")
            if "error" not in ensemble_result:
                print(f"Ensemble Prediction: {ensemble_result['predicted_disease']}")
                print(f"Ensemble Confidence: {ensemble_result['confidence']:.3f}")
            
            # Model comparison
            comparison = ensemble.compare_models(symptom)
            if comparison["agreement"] is not None:
                print(f"Models Agree: {comparison['agreement']}")
                if comparison["agreement"]:
                    print(f"Consensus: {comparison['consensus']['disease']}")
                print(f"Confidence Difference: {comparison['confidence_difference']:.3f}")
            
            print("-" * 50)
    else:
        print("No models could be loaded. Please train the models first.")
        print("Run: python train_ml_model.py && python train_dl_model.py")
