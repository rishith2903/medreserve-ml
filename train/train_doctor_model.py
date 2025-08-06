"""
Train Doctor to Disease and Medicine Model
Maps doctor-entered symptoms to possible diseases and associated medicines
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pickle
import joblib
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_pipeline import MedicalNLPPipeline

class DoctorDiagnosisModel:
    """
    Model to predict diseases and medicines based on doctor-entered symptoms
    """
    
    def __init__(self):
        self.nlp_pipeline = MedicalNLPPipeline()
        self.disease_model = None
        self.medicine_model = None
        self.disease_encoder = None
        self.medicine_encoder = None
        self.feature_names = None
        
        # Medicine mapping for common diseases
        self.disease_medicine_map = {
            'diabetes': ['metformin', 'insulin', 'glipizide', 'pioglitazone'],
            'hypertension': ['lisinopril', 'amlodipine', 'hydrochlorothiazide', 'losartan'],
            'asthma': ['albuterol', 'fluticasone', 'montelukast', 'budesonide'],
            'depression': ['sertraline', 'fluoxetine', 'escitalopram', 'bupropion'],
            'anxiety': ['lorazepam', 'alprazolam', 'buspirone', 'sertraline'],
            'migraine': ['sumatriptan', 'topiramate', 'propranolol', 'amitriptyline'],
            'arthritis': ['ibuprofen', 'naproxen', 'methotrexate', 'prednisone'],
            'gastritis': ['omeprazole', 'ranitidine', 'sucralfate', 'antacids'],
            'pneumonia': ['amoxicillin', 'azithromycin', 'levofloxacin', 'ceftriaxone'],
            'bronchitis': ['azithromycin', 'albuterol', 'prednisone', 'guaifenesin'],
            'uti': ['trimethoprim', 'nitrofurantoin', 'ciprofloxacin', 'amoxicillin'],
            'sinusitis': ['amoxicillin', 'azithromycin', 'fluticasone', 'pseudoephedrine'],
            'eczema': ['hydrocortisone', 'tacrolimus', 'moisturizers', 'antihistamines'],
            'acne': ['benzoyl peroxide', 'tretinoin', 'clindamycin', 'isotretinoin'],
            'gerd': ['omeprazole', 'lansoprazole', 'ranitidine', 'antacids'],
            'osteoporosis': ['alendronate', 'calcium', 'vitamin d', 'risedronate'],
            'thyroid': ['levothyroxine', 'methimazole', 'propylthiouracil', 'iodine'],
            'epilepsy': ['phenytoin', 'carbamazepine', 'valproic acid', 'levetiracetam'],
            'heart failure': ['lisinopril', 'metoprolol', 'furosemide', 'spironolactone'],
            'copd': ['albuterol', 'tiotropium', 'prednisone', 'oxygen therapy']
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data for disease and medicine prediction
        """
        datasets = []
        
        # Load Disease Symptoms and Patient Profile Dataset
        try:
            profile_path = "backend/ml/dataset/Disease Symptoms and Patient Profile Dataset/Disease_symptom_and_patient_profile_dataset.csv"
            if os.path.exists(profile_path):
                df1 = pd.read_csv(profile_path)
                print(f"Loaded Patient Profile dataset: {len(df1)} rows")
                datasets.append(('profile', df1))
        except Exception as e:
            print(f"Error loading Patient Profile dataset: {e}")
        
        # Load Disease-Symptom Dataset
        try:
            disease_symptom_path = "backend/ml/dataset/Disease-Symptom Dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
            if os.path.exists(disease_symptom_path):
                df2 = pd.read_csv(disease_symptom_path)
                print(f"Loaded Disease-Symptom dataset: {len(df2)} rows")
                datasets.append(('disease_symptom', df2))
        except Exception as e:
            print(f"Error loading Disease-Symptom dataset: {e}")
        
        # Load Symptom2Disease dataset
        try:
            symptom2disease_path = "backend/ml/dataset/Symptom2Disease/Symptom2Disease.csv"
            if os.path.exists(symptom2disease_path):
                df3 = pd.read_csv(symptom2disease_path)
                print(f"Loaded Symptom2Disease dataset: {len(df3)} rows")
                datasets.append(('symptom2disease', df3))
        except Exception as e:
            print(f"Error loading Symptom2Disease dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Process and combine datasets
        combined_data = self._process_datasets(datasets)
        
        # Create training data
        symptoms_data, diseases_data, medicines_data = self._create_training_data(combined_data)
        
        return symptoms_data, diseases_data, medicines_data
    
    def _process_datasets(self, datasets: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Process and standardize different dataset formats
        """
        processed_rows = []
        
        for dataset_type, df in datasets:
            print(f"Processing {dataset_type} dataset with columns: {df.columns.tolist()}")
            
            if dataset_type == 'profile':
                # Process patient profile dataset
                for _, row in df.iterrows():
                    if 'Disease' in row and pd.notna(row['Disease']):
                        disease = str(row['Disease']).strip().lower()
                        
                        # Collect symptoms from symptom columns
                        symptoms = []
                        for col in df.columns:
                            if 'symptom' in col.lower() and pd.notna(row[col]):
                                symptom = str(row[col]).strip()
                                if symptom and symptom.lower() not in ['0', 'no', 'none', '']:
                                    symptoms.append(symptom)
                        
                        if symptoms:
                            # Get medicines for this disease
                            medicines = self.disease_medicine_map.get(disease, [])
                            
                            processed_rows.append({
                                'symptoms': ' '.join(symptoms),
                                'disease': disease,
                                'medicines': medicines
                            })
            
            elif dataset_type == 'disease_symptom':
                # Process disease-symptom dataset
                for _, row in df.iterrows():
                    disease_col = None
                    symptom_cols = []
                    
                    # Find disease and symptom columns
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in ['disease', 'diagnosis', 'condition']):
                            disease_col = col
                        elif any(keyword in col_lower for keyword in ['symptom', 'sign']):
                            symptom_cols.append(col)
                    
                    if disease_col and disease_col in row and pd.notna(row[disease_col]):
                        disease = str(row[disease_col]).strip().lower()
                        
                        symptoms = []
                        for col in symptom_cols:
                            if col in row and pd.notna(row[col]):
                                symptom = str(row[col]).strip()
                                if symptom and symptom.lower() not in ['0', 'no', 'none', '']:
                                    symptoms.append(symptom)
                        
                        if symptoms:
                            medicines = self.disease_medicine_map.get(disease, [])
                            
                            processed_rows.append({
                                'symptoms': ' '.join(symptoms),
                                'disease': disease,
                                'medicines': medicines
                            })
            
            elif dataset_type == 'symptom2disease':
                # Process symptom2disease dataset
                for _, row in df.iterrows():
                    if 'label' in row and 'text' in row:
                        disease = str(row['label']).strip().lower()
                        symptoms = str(row['text']).strip()
                        
                        if disease and symptoms:
                            medicines = self.disease_medicine_map.get(disease, [])
                            
                            processed_rows.append({
                                'symptoms': symptoms,
                                'disease': disease,
                                'medicines': medicines
                            })
        
        return pd.DataFrame(processed_rows)
    
    def _create_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create training data for symptoms to diseases and medicines
        """
        # Remove duplicates and clean data
        df = df.drop_duplicates(subset=['symptoms', 'disease'])
        df = df[df['symptoms'].str.len() > 10]  # Filter out very short symptom descriptions
        
        print(f"Created {len(df)} training examples")
        print(f"Unique diseases: {df['disease'].nunique()}")
        
        # Create separate DataFrames
        symptoms_df = pd.DataFrame({'symptoms': df['symptoms']})
        diseases_df = pd.DataFrame({'disease': df['disease']})
        medicines_df = pd.DataFrame({'medicines': df['medicines']})
        
        return symptoms_df, diseases_df, medicines_df
    
    def train_models(self, symptoms_df: pd.DataFrame, diseases_df: pd.DataFrame, medicines_df: pd.DataFrame):
        """
        Train both disease and medicine prediction models
        """
        print("Training Doctor Diagnosis Models...")
        
        # Prepare text data
        symptoms_text = symptoms_df['symptoms'].tolist()
        diseases = diseases_df['disease'].tolist()
        medicines_lists = medicines_df['medicines'].tolist()
        
        # Fit NLP pipeline
        print("Fitting NLP pipeline...")
        self.nlp_pipeline.fit_vectorizer(symptoms_text, max_features=5000)
        
        # Vectorize symptoms
        print("Vectorizing symptoms...")
        X = self.nlp_pipeline.vectorize_text(symptoms_text)
        
        # Prepare disease labels
        print("Encoding diseases...")
        self.disease_encoder = LabelEncoder()
        y_diseases = self.disease_encoder.fit_transform(diseases)
        
        # Prepare medicine labels (multi-label)
        print("Encoding medicines...")
        self.medicine_encoder = MultiLabelBinarizer()
        y_medicines = self.medicine_encoder.fit_transform(medicines_lists)
        
        print(f"Training data shape: {X.shape}")
        print(f"Number of diseases: {len(self.disease_encoder.classes_)}")
        print(f"Number of medicines: {len(self.medicine_encoder.classes_)}")
        
        # Split data
        X_train, X_test, y_diseases_train, y_diseases_test, y_medicines_train, y_medicines_test = train_test_split(
            X, y_diseases, y_medicines, test_size=0.2, random_state=42
        )
        
        # Train disease prediction model
        print("Training disease prediction model...")
        self.disease_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.disease_model.fit(X_train, y_diseases_train)
        
        # Evaluate disease model
        y_diseases_pred = self.disease_model.predict(X_test)
        disease_accuracy = accuracy_score(y_diseases_test, y_diseases_pred)
        print(f"Disease prediction accuracy: {disease_accuracy:.4f}")
        
        # Train medicine prediction model (multi-label)
        print("Training medicine prediction model...")
        self.medicine_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        )
        
        self.medicine_model.fit(X_train, y_medicines_train)
        
        # Evaluate medicine model
        y_medicines_pred = self.medicine_model.predict(X_test)
        medicine_hamming = hamming_loss(y_medicines_test, y_medicines_pred)
        print(f"Medicine prediction hamming loss: {medicine_hamming:.4f}")
        
        # Feature importance for disease model
        feature_names = self.nlp_pipeline.tfidf_vectorizer.get_feature_names_out()
        feature_importance = self.disease_model.feature_importances_
        
        # Get top 15 most important features
        top_features_idx = np.argsort(feature_importance)[-15:]
        print("\nTop 15 Most Important Features for Disease Prediction:")
        for idx in reversed(top_features_idx):
            print(f"- {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        self.feature_names = feature_names
        
        return self.disease_model, self.medicine_model
    
    def predict_diagnosis(self, symptoms_text: str, top_diseases: int = 5, top_medicines: int = 5) -> Dict:
        """
        Predict diseases and medicines for given symptoms
        """
        if not self.disease_model or not self.medicine_model:
            raise ValueError("Models not trained. Call train_models first.")
        
        # Vectorize input symptoms
        X = self.nlp_pipeline.vectorize_text([symptoms_text])
        
        # Predict diseases
        disease_probabilities = self.disease_model.predict_proba(X)[0]
        top_disease_indices = np.argsort(disease_probabilities)[-top_diseases:][::-1]
        
        predicted_diseases = []
        for idx in top_disease_indices:
            disease = self.disease_encoder.classes_[idx]
            probability = disease_probabilities[idx]
            predicted_diseases.append((disease, probability))
        
        # Predict medicines
        medicine_probabilities = self.medicine_model.predict_proba(X)
        
        # Get top medicines across all outputs
        all_medicine_scores = []
        for i, medicine in enumerate(self.medicine_encoder.classes_):
            # Get probability from the corresponding output
            if i < len(medicine_probabilities):
                prob = medicine_probabilities[i][0][1] if len(medicine_probabilities[i][0]) > 1 else 0
                all_medicine_scores.append((medicine, prob))
        
        # Sort and get top medicines
        all_medicine_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_medicines = all_medicine_scores[:top_medicines]
        
        return {
            'diseases': predicted_diseases,
            'medicines': predicted_medicines,
            'symptoms_processed': self.nlp_pipeline.preprocess_text(symptoms_text)
        }
    
    def save_models(self, model_dir: str = "backend/ml/models"):
        """
        Save the trained models and components
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save disease model
        disease_model_path = os.path.join(model_dir, "doctor_disease_model.pkl")
        joblib.dump(self.disease_model, disease_model_path)
        
        # Save medicine model
        medicine_model_path = os.path.join(model_dir, "doctor_medicine_model.pkl")
        joblib.dump(self.medicine_model, medicine_model_path)
        
        # Save NLP pipeline
        nlp_path = os.path.join(model_dir, "doctor_nlp_pipeline.pkl")
        self.nlp_pipeline.save_pipeline(nlp_path)
        
        # Save encoders
        disease_encoder_path = os.path.join(model_dir, "doctor_disease_encoder.pkl")
        joblib.dump(self.disease_encoder, disease_encoder_path)
        
        medicine_encoder_path = os.path.join(model_dir, "doctor_medicine_encoder.pkl")
        joblib.dump(self.medicine_encoder, medicine_encoder_path)
        
        # Save feature names
        features_path = os.path.join(model_dir, "doctor_feature_names.pkl")
        joblib.dump(self.feature_names, features_path)
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "backend/ml/models"):
        """
        Load trained models
        """
        # Load models
        disease_model_path = os.path.join(model_dir, "doctor_disease_model.pkl")
        self.disease_model = joblib.load(disease_model_path)
        
        medicine_model_path = os.path.join(model_dir, "doctor_medicine_model.pkl")
        self.medicine_model = joblib.load(medicine_model_path)
        
        # Load NLP pipeline
        nlp_path = os.path.join(model_dir, "doctor_nlp_pipeline.pkl")
        self.nlp_pipeline.load_pipeline(nlp_path)
        
        # Load encoders
        disease_encoder_path = os.path.join(model_dir, "doctor_disease_encoder.pkl")
        self.disease_encoder = joblib.load(disease_encoder_path)
        
        medicine_encoder_path = os.path.join(model_dir, "doctor_medicine_encoder.pkl")
        self.medicine_encoder = joblib.load(medicine_encoder_path)
        
        # Load feature names
        features_path = os.path.join(model_dir, "doctor_feature_names.pkl")
        self.feature_names = joblib.load(features_path)
        
        print(f"Models loaded from {model_dir}")

def main():
    """
    Main training function
    """
    print("Starting Doctor Diagnosis Model Training...")
    
    # Initialize model
    model = DoctorDiagnosisModel()
    
    # Load and prepare data
    print("Loading and preparing data...")
    symptoms_df, diseases_df, medicines_df = model.load_and_prepare_data()
    
    # Train models
    model.train_models(symptoms_df, diseases_df, medicines_df)
    
    # Save models
    model.save_models()
    
    # Test predictions
    print("\nTesting predictions...")
    test_symptoms = [
        "Patient presents with chest pain, shortness of breath, and sweating",
        "Severe headache with nausea, vomiting, and sensitivity to light",
        "Persistent cough with fever and difficulty breathing",
        "Joint pain and morning stiffness in hands and feet",
        "Abdominal pain with nausea and loss of appetite"
    ]
    
    for symptoms in test_symptoms:
        diagnosis = model.predict_diagnosis(symptoms, top_diseases=3, top_medicines=3)
        print(f"\nSymptoms: {symptoms}")
        print("Predicted diseases:")
        for disease, prob in diagnosis['diseases']:
            print(f"  - {disease}: {prob:.3f}")
        print("Recommended medicines:")
        for medicine, prob in diagnosis['medicines']:
            print(f"  - {medicine}: {prob:.3f}")

if __name__ == "__main__":
    main()
