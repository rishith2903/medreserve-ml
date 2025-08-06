"""
Patient to Doctor Specialization Prediction
Predicts recommended doctor specializations based on patient symptoms
"""

import os
import sys
import joblib
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_pipeline import MedicalNLPPipeline
from utils.mapping_specialization import DiseaseSpecializationMapper

class SpecializationPredictor:
    """
    Predicts doctor specializations based on patient symptoms
    """
    
    def __init__(self, model_dir: str = "backend/ml/models"):
        self.model_dir = model_dir
        self.nlp_pipeline = MedicalNLPPipeline()
        self.disease_mapper = DiseaseSpecializationMapper()
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load the trained patient to specialization model
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, "patient_to_specialization_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            
            # Load NLP pipeline
            nlp_path = os.path.join(self.model_dir, "patient_nlp_pipeline.pkl")
            if os.path.exists(nlp_path):
                self.nlp_pipeline.load_pipeline(nlp_path)
            else:
                print("Warning: NLP pipeline not found, using default")
            
            # Load label encoder
            encoder_path = os.path.join(self.model_dir, "patient_label_encoder.pkl")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
            else:
                raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, "patient_feature_names.pkl")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            self.is_loaded = True
            print("Patient specialization model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            raise
    
    def predict_specializations(self, symptoms_text: str, top_k: int = 3) -> Dict:
        """
        Predict top K doctor specializations for given symptoms
        
        Args:
            symptoms_text (str): Patient's symptom description
            top_k (int): Number of top specializations to return
            
        Returns:
            Dict: Prediction results with specializations, confidence scores, and metadata
        """
        if not self.is_loaded:
            self.load_model()
        
        if not symptoms_text or not symptoms_text.strip():
            return {
                'specializations': [],
                'confidence': 0.0,
                'processed_symptoms': '',
                'error': 'Empty symptoms text provided'
            }
        
        try:
            # Preprocess symptoms
            processed_symptoms = self.nlp_pipeline.preprocess_text(symptoms_text)
            
            # Vectorize input symptoms
            X = self.nlp_pipeline.vectorize_text([symptoms_text])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Get top K predictions
            top_indices = probabilities.argsort()[-top_k:][::-1]
            
            predictions = []
            total_confidence = 0.0
            
            for idx in top_indices:
                specialization = self.label_encoder.classes_[idx]
                probability = float(probabilities[idx])
                total_confidence += probability
                
                predictions.append({
                    'specialization': specialization,
                    'confidence': probability,
                    'percentage': probability * 100
                })
            
            # Calculate overall confidence
            overall_confidence = total_confidence / top_k if top_k > 0 else 0.0
            
            # Add additional recommendations based on keyword matching
            keyword_recommendations = self._get_keyword_based_recommendations(symptoms_text)
            
            return {
                'specializations': predictions,
                'confidence': overall_confidence,
                'processed_symptoms': processed_symptoms,
                'keyword_recommendations': keyword_recommendations,
                'model_version': 'v1.0',
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            return {
                'specializations': [],
                'confidence': 0.0,
                'processed_symptoms': '',
                'error': f'Prediction error: {str(e)}'
            }
    
    def _get_keyword_based_recommendations(self, symptoms_text: str) -> List[Dict]:
        """
        Get additional recommendations based on keyword matching
        """
        symptoms_lower = symptoms_text.lower()
        keyword_recommendations = []
        
        # Define keyword-to-specialization mappings
        keyword_mappings = {
            'chest pain': 'Cardiology',
            'heart': 'Cardiology',
            'cardiac': 'Cardiology',
            'blood pressure': 'Cardiology',
            'headache': 'Neurology',
            'migraine': 'Neurology',
            'seizure': 'Neurology',
            'stroke': 'Neurology',
            'breathing': 'Pulmonology',
            'cough': 'Pulmonology',
            'asthma': 'Pulmonology',
            'lung': 'Pulmonology',
            'stomach': 'Gastroenterology',
            'abdominal': 'Gastroenterology',
            'nausea': 'Gastroenterology',
            'vomiting': 'Gastroenterology',
            'skin': 'Dermatology',
            'rash': 'Dermatology',
            'eczema': 'Dermatology',
            'acne': 'Dermatology',
            'joint': 'Orthopedics',
            'bone': 'Orthopedics',
            'fracture': 'Orthopedics',
            'back pain': 'Orthopedics',
            'depression': 'Psychiatry',
            'anxiety': 'Psychiatry',
            'mental': 'Psychiatry',
            'mood': 'Psychiatry',
            'eye': 'Ophthalmology',
            'vision': 'Ophthalmology',
            'sight': 'Ophthalmology',
            'ear': 'ENT',
            'hearing': 'ENT',
            'throat': 'ENT',
            'sinus': 'ENT'
        }
        
        for keyword, specialization in keyword_mappings.items():
            if keyword in symptoms_lower:
                keyword_recommendations.append({
                    'specialization': specialization,
                    'reason': f"Keyword match: '{keyword}'",
                    'confidence': 0.8
                })
        
        # Remove duplicates and limit to top 3
        seen_specs = set()
        unique_recommendations = []
        for rec in keyword_recommendations:
            if rec['specialization'] not in seen_specs:
                seen_specs.add(rec['specialization'])
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= 3:
                    break
        
        return unique_recommendations
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'model_type': 'Random Forest Classifier',
            'specializations': list(self.label_encoder.classes_) if self.label_encoder else [],
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'model_dir': self.model_dir,
            'is_loaded': self.is_loaded
        }
    
    def batch_predict(self, symptoms_list: List[str], top_k: int = 3) -> List[Dict]:
        """
        Predict specializations for multiple symptom descriptions
        """
        results = []
        for symptoms in symptoms_list:
            result = self.predict_specializations(symptoms, top_k)
            results.append(result)
        return results

# Fallback prediction function for when model is not available
def fallback_specialization_prediction(symptoms_text: str, top_k: int = 3) -> Dict:
    """
    Fallback prediction using rule-based approach when ML model is not available
    """
    symptoms_lower = symptoms_text.lower()
    
    # Rule-based specialization scoring
    specialization_scores = {
        'Internal Medicine': 1.0,  # Default baseline
        'Cardiology': 0.0,
        'Neurology': 0.0,
        'Pulmonology': 0.0,
        'Gastroenterology': 0.0,
        'Dermatology': 0.0,
        'Orthopedics': 0.0,
        'Psychiatry': 0.0,
        'Ophthalmology': 0.0,
        'ENT': 0.0
    }
    
    # Cardiology keywords
    if any(keyword in symptoms_lower for keyword in ['chest pain', 'heart', 'cardiac', 'blood pressure', 'palpitation']):
        specialization_scores['Cardiology'] += 3.0
    
    # Neurology keywords
    if any(keyword in symptoms_lower for keyword in ['headache', 'migraine', 'seizure', 'dizziness', 'stroke']):
        specialization_scores['Neurology'] += 3.0
    
    # Pulmonology keywords
    if any(keyword in symptoms_lower for keyword in ['breathing', 'cough', 'asthma', 'lung', 'shortness']):
        specialization_scores['Pulmonology'] += 3.0
    
    # Gastroenterology keywords
    if any(keyword in symptoms_lower for keyword in ['stomach', 'abdominal', 'nausea', 'vomiting', 'diarrhea']):
        specialization_scores['Gastroenterology'] += 3.0
    
    # Dermatology keywords
    if any(keyword in symptoms_lower for keyword in ['skin', 'rash', 'eczema', 'acne', 'itching']):
        specialization_scores['Dermatology'] += 3.0
    
    # Orthopedics keywords
    if any(keyword in symptoms_lower for keyword in ['joint', 'bone', 'fracture', 'back pain', 'arthritis']):
        specialization_scores['Orthopedics'] += 3.0
    
    # Psychiatry keywords
    if any(keyword in symptoms_lower for keyword in ['depression', 'anxiety', 'mental', 'mood', 'stress']):
        specialization_scores['Psychiatry'] += 3.0
    
    # Sort by score and get top K
    sorted_specs = sorted(specialization_scores.items(), key=lambda x: x[1], reverse=True)
    
    predictions = []
    for i, (spec, score) in enumerate(sorted_specs[:top_k]):
        if score > 0:
            confidence = min(score / 5.0, 1.0)  # Normalize to 0-1
            predictions.append({
                'specialization': spec,
                'confidence': confidence,
                'percentage': confidence * 100
            })
    
    # If no specific matches, recommend Internal Medicine
    if not predictions:
        predictions.append({
            'specialization': 'Internal Medicine',
            'confidence': 0.7,
            'percentage': 70.0
        })
    
    return {
        'specializations': predictions,
        'confidence': predictions[0]['confidence'] if predictions else 0.0,
        'processed_symptoms': symptoms_text.lower().strip(),
        'model_version': 'fallback_v1.0',
        'note': 'Using rule-based fallback prediction'
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    predictor = SpecializationPredictor()
    
    try:
        predictor.load_model()
        
        # Test symptoms
        test_symptoms = [
            "I have severe chest pain and shortness of breath",
            "Experiencing headache, dizziness and nausea for 3 days",
            "Skin rash with itching and redness on arms",
            "Joint pain and stiffness in the morning",
            "Feeling sad, anxious and having trouble sleeping"
        ]
        
        print("Testing Patient to Specialization Predictions:")
        print("=" * 60)
        
        for symptoms in test_symptoms:
            result = predictor.predict_specializations(symptoms, top_k=3)
            
            print(f"\nSymptoms: {symptoms}")
            print("Recommended specializations:")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                for spec_info in result['specializations']:
                    print(f"  - {spec_info['specialization']}: {spec_info['percentage']:.1f}%")
                
                if result.get('keyword_recommendations'):
                    print("Additional keyword-based recommendations:")
                    for rec in result['keyword_recommendations']:
                        print(f"  - {rec['specialization']}: {rec['reason']}")
        
        # Test model info
        print(f"\nModel Info: {predictor.get_model_info()}")
        
    except Exception as e:
        print(f"Model not available, testing fallback prediction: {e}")
        
        # Test fallback prediction
        test_symptoms = "I have severe chest pain and shortness of breath"
        result = fallback_specialization_prediction(test_symptoms)
        
        print(f"\nFallback prediction for: {test_symptoms}")
        for spec_info in result['specializations']:
            print(f"  - {spec_info['specialization']}: {spec_info['percentage']:.1f}%")
