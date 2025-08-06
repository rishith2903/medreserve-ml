"""
Doctor to Disease and Medicine Prediction
Predicts possible diseases and associated medicines based on doctor-entered symptoms
"""

import os
import sys
import joblib
import numpy as np
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_pipeline import MedicalNLPPipeline

class DiseaseMedicinePredictor:
    """
    Predicts diseases and medicines based on doctor-entered symptoms
    """
    
    def __init__(self, model_dir: str = "backend/ml/models"):
        self.model_dir = model_dir
        self.nlp_pipeline = MedicalNLPPipeline()
        self.disease_model = None
        self.medicine_model = None
        self.disease_encoder = None
        self.medicine_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        # Comprehensive disease-medicine mapping for fallback
        self.disease_medicine_map = {
            'diabetes': ['metformin', 'insulin', 'glipizide', 'pioglitazone', 'sitagliptin'],
            'hypertension': ['lisinopril', 'amlodipine', 'hydrochlorothiazide', 'losartan', 'metoprolol'],
            'asthma': ['albuterol', 'fluticasone', 'montelukast', 'budesonide', 'theophylline'],
            'depression': ['sertraline', 'fluoxetine', 'escitalopram', 'bupropion', 'venlafaxine'],
            'anxiety': ['lorazepam', 'alprazolam', 'buspirone', 'sertraline', 'clonazepam'],
            'migraine': ['sumatriptan', 'topiramate', 'propranolol', 'amitriptyline', 'rizatriptan'],
            'arthritis': ['ibuprofen', 'naproxen', 'methotrexate', 'prednisone', 'celecoxib'],
            'gastritis': ['omeprazole', 'ranitidine', 'sucralfate', 'antacids', 'lansoprazole'],
            'pneumonia': ['amoxicillin', 'azithromycin', 'levofloxacin', 'ceftriaxone', 'doxycycline'],
            'bronchitis': ['azithromycin', 'albuterol', 'prednisone', 'guaifenesin', 'dextromethorphan'],
            'uti': ['trimethoprim', 'nitrofurantoin', 'ciprofloxacin', 'amoxicillin', 'fosfomycin'],
            'sinusitis': ['amoxicillin', 'azithromycin', 'fluticasone', 'pseudoephedrine', 'saline rinse'],
            'eczema': ['hydrocortisone', 'tacrolimus', 'moisturizers', 'antihistamines', 'pimecrolimus'],
            'acne': ['benzoyl peroxide', 'tretinoin', 'clindamycin', 'isotretinoin', 'adapalene'],
            'gerd': ['omeprazole', 'lansoprazole', 'ranitidine', 'antacids', 'esomeprazole'],
            'osteoporosis': ['alendronate', 'calcium', 'vitamin d', 'risedronate', 'denosumab'],
            'thyroid': ['levothyroxine', 'methimazole', 'propylthiouracil', 'iodine', 'liothyronine'],
            'epilepsy': ['phenytoin', 'carbamazepine', 'valproic acid', 'levetiracetam', 'lamotrigine'],
            'heart failure': ['lisinopril', 'metoprolol', 'furosemide', 'spironolactone', 'digoxin'],
            'copd': ['albuterol', 'tiotropium', 'prednisone', 'oxygen therapy', 'budesonide'],
            'insomnia': ['zolpidem', 'melatonin', 'trazodone', 'eszopiclone', 'diphenhydramine'],
            'allergies': ['loratadine', 'cetirizine', 'fexofenadine', 'benadryl', 'fluticasone'],
            'constipation': ['docusate', 'polyethylene glycol', 'bisacodyl', 'senna', 'lactulose'],
            'diarrhea': ['loperamide', 'bismuth subsalicylate', 'probiotics', 'oral rehydration', 'kaolin'],
            'nausea': ['ondansetron', 'metoclopramide', 'promethazine', 'ginger', 'dimenhydrinate'],
            'fever': ['acetaminophen', 'ibuprofen', 'aspirin', 'naproxen', 'cooling measures']
        }
    
    def load_models(self):
        """
        Load the trained disease and medicine prediction models
        """
        try:
            # Load disease model
            disease_model_path = os.path.join(self.model_dir, "doctor_disease_model.pkl")
            if os.path.exists(disease_model_path):
                self.disease_model = joblib.load(disease_model_path)
            
            # Load medicine model
            medicine_model_path = os.path.join(self.model_dir, "doctor_medicine_model.pkl")
            if os.path.exists(medicine_model_path):
                self.medicine_model = joblib.load(medicine_model_path)
            
            # Load NLP pipeline
            nlp_path = os.path.join(self.model_dir, "doctor_nlp_pipeline.pkl")
            if os.path.exists(nlp_path):
                self.nlp_pipeline.load_pipeline(nlp_path)
            
            # Load encoders
            disease_encoder_path = os.path.join(self.model_dir, "doctor_disease_encoder.pkl")
            if os.path.exists(disease_encoder_path):
                self.disease_encoder = joblib.load(disease_encoder_path)
            
            medicine_encoder_path = os.path.join(self.model_dir, "doctor_medicine_encoder.pkl")
            if os.path.exists(medicine_encoder_path):
                self.medicine_encoder = joblib.load(medicine_encoder_path)
            
            # Load feature names
            features_path = os.path.join(self.model_dir, "doctor_feature_names.pkl")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            self.is_loaded = (self.disease_model is not None and 
                            self.disease_encoder is not None)
            
            if self.is_loaded:
                print("Doctor diagnosis models loaded successfully")
            else:
                print("Warning: Some model components could not be loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
    
    def predict_diagnosis(self, symptoms_text: str, top_diseases: int = 5, top_medicines: int = 5) -> Dict:
        """
        Predict diseases and medicines for given symptoms
        
        Args:
            symptoms_text (str): Doctor's symptom description
            top_diseases (int): Number of top diseases to return
            top_medicines (int): Number of top medicines to return
            
        Returns:
            Dict: Prediction results with diseases, medicines, and metadata
        """
        if not self.is_loaded:
            self.load_models()
        
        if not symptoms_text or not symptoms_text.strip():
            return {
                'diseases': [],
                'medicines': [],
                'confidence': 0.0,
                'processed_symptoms': '',
                'error': 'Empty symptoms text provided'
            }
        
        # If models are not available, use fallback prediction
        if not self.is_loaded:
            return self._fallback_prediction(symptoms_text, top_diseases, top_medicines)
        
        try:
            # Preprocess symptoms
            processed_symptoms = self.nlp_pipeline.preprocess_text(symptoms_text)
            
            # Vectorize input symptoms
            X = self.nlp_pipeline.vectorize_text([symptoms_text])
            
            # Predict diseases
            disease_probabilities = self.disease_model.predict_proba(X)[0]
            top_disease_indices = np.argsort(disease_probabilities)[-top_diseases:][::-1]
            
            predicted_diseases = []
            for idx in top_disease_indices:
                disease = self.disease_encoder.classes_[idx]
                probability = float(disease_probabilities[idx])
                predicted_diseases.append({
                    'disease': disease,
                    'confidence': probability,
                    'percentage': probability * 100
                })
            
            # Predict medicines (if medicine model is available)
            predicted_medicines = []
            if self.medicine_model and self.medicine_encoder:
                try:
                    medicine_probabilities = self.medicine_model.predict_proba(X)
                    
                    # Get top medicines across all outputs
                    all_medicine_scores = []
                    for i, medicine in enumerate(self.medicine_encoder.classes_):
                        if i < len(medicine_probabilities):
                            prob = medicine_probabilities[i][0][1] if len(medicine_probabilities[i][0]) > 1 else 0
                            all_medicine_scores.append((medicine, float(prob)))
                    
                    # Sort and get top medicines
                    all_medicine_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    for medicine, prob in all_medicine_scores[:top_medicines]:
                        predicted_medicines.append({
                            'medicine': medicine,
                            'confidence': prob,
                            'percentage': prob * 100
                        })
                        
                except Exception as e:
                    print(f"Error predicting medicines: {e}")
                    # Fallback to disease-based medicine recommendation
                    predicted_medicines = self._get_medicines_for_diseases(predicted_diseases, top_medicines)
            else:
                # Use disease-based medicine recommendation
                predicted_medicines = self._get_medicines_for_diseases(predicted_diseases, top_medicines)
            
            # Calculate overall confidence
            overall_confidence = np.mean([d['confidence'] for d in predicted_diseases]) if predicted_diseases else 0.0
            
            return {
                'diseases': predicted_diseases,
                'medicines': predicted_medicines,
                'confidence': overall_confidence,
                'processed_symptoms': processed_symptoms,
                'model_version': 'v1.0',
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            return {
                'diseases': [],
                'medicines': [],
                'confidence': 0.0,
                'processed_symptoms': '',
                'error': f'Prediction error: {str(e)}'
            }
    
    def _get_medicines_for_diseases(self, predicted_diseases: List[Dict], top_medicines: int) -> List[Dict]:
        """
        Get medicines based on predicted diseases using the disease-medicine mapping
        """
        medicine_scores = {}
        
        for disease_info in predicted_diseases:
            disease = disease_info['disease'].lower()
            confidence = disease_info['confidence']
            
            # Find medicines for this disease
            medicines = self.disease_medicine_map.get(disease, [])
            
            for medicine in medicines:
                if medicine in medicine_scores:
                    medicine_scores[medicine] += confidence
                else:
                    medicine_scores[medicine] = confidence
        
        # Sort by score and return top medicines
        sorted_medicines = sorted(medicine_scores.items(), key=lambda x: x[1], reverse=True)
        
        predicted_medicines = []
        for medicine, score in sorted_medicines[:top_medicines]:
            predicted_medicines.append({
                'medicine': medicine,
                'confidence': min(score, 1.0),
                'percentage': min(score * 100, 100.0)
            })
        
        return predicted_medicines
    
    def _fallback_prediction(self, symptoms_text: str, top_diseases: int, top_medicines: int) -> Dict:
        """
        Fallback prediction using rule-based approach when ML models are not available
        """
        symptoms_lower = symptoms_text.lower()
        
        # Rule-based disease scoring
        disease_scores = {}
        
        # Define symptom-to-disease mappings
        symptom_disease_map = {
            'chest pain': ['heart attack', 'angina', 'pneumonia', 'gerd'],
            'shortness of breath': ['asthma', 'heart failure', 'pneumonia', 'copd'],
            'headache': ['migraine', 'tension headache', 'sinusitis', 'hypertension'],
            'fever': ['infection', 'pneumonia', 'flu', 'bronchitis'],
            'cough': ['bronchitis', 'pneumonia', 'asthma', 'copd'],
            'nausea': ['gastritis', 'food poisoning', 'migraine', 'pregnancy'],
            'vomiting': ['gastritis', 'food poisoning', 'migraine', 'appendicitis'],
            'abdominal pain': ['gastritis', 'appendicitis', 'gallstones', 'ulcer'],
            'joint pain': ['arthritis', 'gout', 'lupus', 'fibromyalgia'],
            'skin rash': ['eczema', 'allergies', 'psoriasis', 'dermatitis'],
            'dizziness': ['vertigo', 'hypotension', 'anemia', 'dehydration'],
            'fatigue': ['anemia', 'depression', 'thyroid', 'diabetes'],
            'back pain': ['muscle strain', 'herniated disc', 'arthritis', 'kidney stones'],
            'sore throat': ['strep throat', 'viral infection', 'tonsillitis', 'allergies']
        }
        
        # Score diseases based on symptom matches
        for symptom, diseases in symptom_disease_map.items():
            if symptom in symptoms_lower:
                for disease in diseases:
                    disease_scores[disease] = disease_scores.get(disease, 0) + 1
        
        # If no specific matches, add common conditions
        if not disease_scores:
            disease_scores = {
                'viral infection': 0.5,
                'bacterial infection': 0.4,
                'allergic reaction': 0.3,
                'stress-related': 0.2
            }
        
        # Sort and get top diseases
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        
        predicted_diseases = []
        for disease, score in sorted_diseases[:top_diseases]:
            confidence = min(score / 3.0, 1.0)  # Normalize
            predicted_diseases.append({
                'disease': disease,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        # Get medicines for predicted diseases
        predicted_medicines = self._get_medicines_for_diseases(predicted_diseases, top_medicines)
        
        return {
            'diseases': predicted_diseases,
            'medicines': predicted_medicines,
            'confidence': predicted_diseases[0]['confidence'] if predicted_diseases else 0.0,
            'processed_symptoms': symptoms_text.lower().strip(),
            'model_version': 'fallback_v1.0',
            'note': 'Using rule-based fallback prediction'
        }
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models
        """
        return {
            'disease_model_loaded': self.disease_model is not None,
            'medicine_model_loaded': self.medicine_model is not None,
            'diseases': list(self.disease_encoder.classes_) if self.disease_encoder else [],
            'medicines': list(self.medicine_encoder.classes_) if self.medicine_encoder else [],
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'model_dir': self.model_dir,
            'is_loaded': self.is_loaded
        }
    
    def batch_predict(self, symptoms_list: List[str], top_diseases: int = 5, top_medicines: int = 5) -> List[Dict]:
        """
        Predict diagnosis for multiple symptom descriptions
        """
        results = []
        for symptoms in symptoms_list:
            result = self.predict_diagnosis(symptoms, top_diseases, top_medicines)
            results.append(result)
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    predictor = DiseaseMedicinePredictor()
    
    # Test symptoms
    test_symptoms = [
        "Patient presents with chest pain, shortness of breath, and sweating",
        "Severe headache with nausea, vomiting, and sensitivity to light",
        "Persistent cough with fever and difficulty breathing",
        "Joint pain and morning stiffness in hands and feet",
        "Abdominal pain with nausea and loss of appetite"
    ]
    
    print("Testing Doctor to Disease/Medicine Predictions:")
    print("=" * 60)
    
    for symptoms in test_symptoms:
        result = predictor.predict_diagnosis(symptoms, top_diseases=3, top_medicines=3)
        
        print(f"\nSymptoms: {symptoms}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("Predicted diseases:")
            for disease_info in result['diseases']:
                print(f"  - {disease_info['disease']}: {disease_info['percentage']:.1f}%")
            
            print("Recommended medicines:")
            for medicine_info in result['medicines']:
                print(f"  - {medicine_info['medicine']}: {medicine_info['percentage']:.1f}%")
            
            if 'note' in result:
                print(f"Note: {result['note']}")
    
    # Test model info
    print(f"\nModel Info: {predictor.get_model_info()}")
