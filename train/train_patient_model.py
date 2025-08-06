"""
Train Patient to Doctor Specialization Model
Maps patient symptoms to recommended doctor specializations
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.nlp_pipeline import MedicalNLPPipeline
from utils.mapping_specialization import DiseaseSpecializationMapper

class PatientSpecializationModel:
    """
    Model to predict doctor specializations based on patient symptoms
    """
    
    def __init__(self):
        self.nlp_pipeline = MedicalNLPPipeline()
        self.disease_mapper = DiseaseSpecializationMapper()
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data from available datasets
        """
        datasets = []
        
        # Load Disease-Symptom Dataset
        try:
            disease_symptom_path = "backend/ml/dataset/Disease-Symptom Dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
            if os.path.exists(disease_symptom_path):
                df1 = pd.read_csv(disease_symptom_path)
                print(f"Loaded Disease-Symptom dataset: {len(df1)} rows")
                datasets.append(df1)
        except Exception as e:
            print(f"Error loading Disease-Symptom dataset: {e}")

        # Load Doctor's Specialty Recommendation dataset
        try:
            doctor_disease_path = "backend/ml/dataset/Doctor's Specialty Recommendation/Doctor_Versus_Disease.csv"
            if os.path.exists(doctor_disease_path):
                df2 = pd.read_csv(doctor_disease_path)
                print(f"Loaded Doctor-Disease dataset: {len(df2)} rows")
                datasets.append(df2)
        except Exception as e:
            print(f"Error loading Doctor-Disease dataset: {e}")

        # Load Symptom2Disease dataset
        try:
            symptom2disease_path = "backend/ml/dataset/Symptom2Disease/Symptom2Disease.csv"
            if os.path.exists(symptom2disease_path):
                df3 = pd.read_csv(symptom2disease_path)
                print(f"Loaded Symptom2Disease dataset: {len(df3)} rows")
                datasets.append(df3)
        except Exception as e:
            print(f"Error loading Symptom2Disease dataset: {e}")

        # Load sample dataset
        try:
            sample_path = "backend/ml/dataset/sample_symptoms_dataset.csv"
            if os.path.exists(sample_path):
                df4 = pd.read_csv(sample_path)
                print(f"Loaded sample dataset: {len(df4)} rows")
                datasets.append(df4)
        except Exception as e:
            print(f"Error loading sample dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Combine and standardize datasets
        combined_data = self._standardize_datasets(datasets)
        
        # Create symptoms and specializations
        symptoms_data, specializations_data = self._create_training_data(combined_data)
        
        return symptoms_data, specializations_data
    
    def _standardize_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Standardize different dataset formats into a common format
        """
        standardized_rows = []
        
        for df in datasets:
            print(f"Processing dataset with columns: {df.columns.tolist()}")
            
            # Try to identify disease and symptom columns
            disease_col = None
            symptom_cols = []
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['disease', 'diagnosis', 'condition']):
                    disease_col = col
                elif any(keyword in col_lower for keyword in ['symptom', 'sign', 'feature']):
                    symptom_cols.append(col)
            
            # If no specific symptom columns found, use all non-disease columns
            if not symptom_cols and disease_col:
                symptom_cols = [col for col in df.columns if col != disease_col]
            
            # Process each row
            for _, row in df.iterrows():
                if disease_col and disease_col in row:
                    disease = str(row[disease_col]).strip()
                    
                    # Collect symptoms from symptom columns
                    symptoms = []
                    for col in symptom_cols:
                        if col in row and pd.notna(row[col]):
                            value = str(row[col]).strip()
                            if value and value.lower() not in ['0', 'false', 'no', 'none', '']:
                                symptoms.append(value)
                    
                    if disease and symptoms:
                        # Get specialization for this disease
                        specialization = self.disease_mapper.get_specialization(disease)
                        
                        standardized_rows.append({
                            'disease': disease,
                            'symptoms': ' '.join(symptoms),
                            'specialization': specialization
                        })
        
        return pd.DataFrame(standardized_rows)
    
    def _create_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training data for symptoms to specialization mapping
        """
        # Group by specialization and collect symptoms
        specialization_symptoms = {}
        
        for _, row in df.iterrows():
            spec = row['specialization']
            symptoms = row['symptoms']
            
            if spec not in specialization_symptoms:
                specialization_symptoms[spec] = []
            specialization_symptoms[spec].append(symptoms)
        
        # Create training examples
        training_data = []
        for spec, symptom_list in specialization_symptoms.items():
            for symptoms in symptom_list:
                training_data.append({
                    'symptoms': symptoms,
                    'specialization': spec
                })
        
        # Convert to DataFrames
        training_df = pd.DataFrame(training_data)
        
        # Create separate DataFrames for symptoms and specializations
        symptoms_df = pd.DataFrame({'symptoms': training_df['symptoms']})
        specializations_df = pd.DataFrame({'specialization': training_df['specialization']})
        
        print(f"Created {len(training_df)} training examples")
        print(f"Specializations: {specializations_df['specialization'].unique()}")
        
        return symptoms_df, specializations_df
    
    def train_model(self, symptoms_df: pd.DataFrame, specializations_df: pd.DataFrame):
        """
        Train the patient to specialization model
        """
        print("Training Patient to Specialization Model...")
        
        # Prepare text data
        symptoms_text = symptoms_df['symptoms'].tolist()
        specializations = specializations_df['specialization'].tolist()
        
        # Fit NLP pipeline
        print("Fitting NLP pipeline...")
        self.nlp_pipeline.fit_vectorizer(symptoms_text, max_features=3000)
        
        # Vectorize symptoms
        print("Vectorizing symptoms...")
        X = self.nlp_pipeline.vectorize_text(symptoms_text)
        
        # Encode specializations
        print("Encoding specializations...")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(specializations)
        
        print(f"Training data shape: {X.shape}")
        print(f"Number of specializations: {len(self.label_encoder.classes_)}")
        print(f"Specializations: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model with hyperparameter tuning
        print("Training Random Forest model...")
        
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Use a smaller grid for faster training
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=3, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        feature_names = self.nlp_pipeline.tfidf_vectorizer.get_feature_names_out()
        feature_importance = self.model.feature_importances_
        
        # Get top 20 most important features
        top_features_idx = np.argsort(feature_importance)[-20:]
        print("\nTop 20 Most Important Features:")
        for idx in reversed(top_features_idx):
            print(f"- {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        self.feature_names = feature_names
        
        return accuracy

    def train_model_simple(self, X_data, y_data) -> float:
        """
        Train the patient to specialization model with simple X, y format
        """
        print("Training Patient to Specialization Model (Simple)...")

        # Convert to lists if needed
        if hasattr(X_data, 'tolist'):
            symptoms_text = X_data.tolist()
        else:
            symptoms_text = list(X_data)

        if hasattr(y_data, 'tolist'):
            specializations = y_data.tolist()
        else:
            specializations = list(y_data)

        # Fit NLP pipeline
        print("Fitting NLP pipeline...")
        self.nlp_pipeline.fit_vectorizer(symptoms_text, max_features=3000)

        # Vectorize symptoms
        print("Vectorizing symptoms...")
        X = self.nlp_pipeline.vectorize_text(symptoms_text)

        # Encode specializations
        print("Encoding specializations...")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(specializations)

        print(f"Training data shape: {X.shape}")
        print(f"Number of specializations: {len(self.label_encoder.classes_)}")
        print(f"Specializations: {self.label_encoder.classes_}")

        # Split data (handle small datasets)
        n_classes = len(self.label_encoder.classes_)
        n_samples = len(y)

        if n_samples < n_classes * 2:
            # Too few samples for stratified split, use simple split
            test_size = max(0.1, min(0.3, n_classes / n_samples))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            # Enough samples for stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Train Random Forest model (simplified for faster training)
        print("Training Random Forest model...")

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test accuracy: {accuracy:.4f}")

        # Detailed classification report
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Feature importance
        feature_names = self.nlp_pipeline.tfidf_vectorizer.get_feature_names_out()
        self.feature_names = feature_names

        return accuracy

    def predict_specialization(self, symptoms_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict top K specializations for given symptoms
        """
        if not self.model or not self.nlp_pipeline.tfidf_vectorizer:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Vectorize input symptoms
        X = self.nlp_pipeline.vectorize_text([symptoms_text])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top K predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            specialization = self.label_encoder.classes_[idx]
            probability = probabilities[idx]
            predictions.append((specialization, probability))
        
        return predictions
    
    def save_model(self, model_dir: str = "../models"):
        """
        Save the trained model and components
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "patient_to_specialization_model.pkl")
        joblib.dump(self.model, model_path)
        
        # Save NLP pipeline
        nlp_path = os.path.join(model_dir, "patient_nlp_pipeline.pkl")
        self.nlp_pipeline.save_pipeline(nlp_path)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, "patient_label_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names
        features_path = os.path.join(model_dir, "patient_feature_names.pkl")
        joblib.dump(self.feature_names, features_path)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str = "../models"):
        """
        Load a trained model
        """
        # Load model
        model_path = os.path.join(model_dir, "patient_to_specialization_model.pkl")
        self.model = joblib.load(model_path)
        
        # Load NLP pipeline
        nlp_path = os.path.join(model_dir, "patient_nlp_pipeline.pkl")
        self.nlp_pipeline.load_pipeline(nlp_path)
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, "patient_label_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)
        
        # Load feature names
        features_path = os.path.join(model_dir, "patient_feature_names.pkl")
        self.feature_names = joblib.load(features_path)
        
        print(f"Model loaded from {model_dir}")

def main():
    """
    Main training function
    """
    print("Starting Patient to Specialization Model Training...")
    
    # Initialize model
    model = PatientSpecializationModel()
    
    # Load and prepare data
    print("Loading and preparing data...")
    symptoms_df, specializations_df = model.load_and_prepare_data()
    
    # Train model
    model.train_model(symptoms_df, specializations_df)
    
    # Save model
    model.save_model()
    
    # Test predictions
    print("\nTesting predictions...")
    test_symptoms = [
        "I have severe chest pain and shortness of breath",
        "Experiencing headache, dizziness and nausea for 3 days",
        "Skin rash with itching and redness",
        "Joint pain and stiffness in the morning",
        "Feeling sad, anxious and having trouble sleeping"
    ]
    
    for symptoms in test_symptoms:
        predictions = model.predict_specialization(symptoms, top_k=3)
        print(f"\nSymptoms: {symptoms}")
        print("Recommended specializations:")
        for spec, prob in predictions:
            print(f"  - {spec}: {prob:.3f}")

if __name__ == "__main__":
    main()
