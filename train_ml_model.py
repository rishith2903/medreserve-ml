"""
Machine Learning model training script for disease prediction.
Uses TF-IDF features with RandomForestClassifier.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle
from preprocessing.preprocess import TextPreprocessor, save_preprocessor
import matplotlib.pyplot as plt
import seaborn as sns

class MLDiseasePredictor:
    """Machine Learning based disease prediction using TF-IDF + Random Forest."""
    
    def __init__(self, data_path: str = "data/symptoms_dataset.csv"):
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_columns = ['symptoms', 'disease']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        # Remove any rows with missing values
        df = df.dropna(subset=required_columns)
        
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Number of unique diseases: {df['disease'].nunique()}")
        print(f"Disease distribution:\n{df['disease'].value_counts().head(10)}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the text data and encode labels."""
        print("Preprocessing text data...")
        
        # Clean symptom descriptions
        cleaned_symptoms = []
        for symptom in df['symptoms']:
            cleaned = self.preprocessor.clean_text(str(symptom))
            cleaned_symptoms.append(cleaned)
        
        # Remove empty cleaned texts
        valid_indices = [i for i, text in enumerate(cleaned_symptoms) if text.strip()]
        cleaned_symptoms = [cleaned_symptoms[i] for i in valid_indices]
        diseases = df['disease'].iloc[valid_indices].tolist()
        
        print(f"After cleaning: {len(cleaned_symptoms)} valid samples")
        
        # Convert to TF-IDF features
        X, self.vectorizer = self.preprocessor.get_tfidf_features(cleaned_symptoms, max_features=5000)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Encode disease labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(diseases)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X, y, cleaned_symptoms, diseases
    
    def train_model(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """Train the Random Forest model with hyperparameter tuning."""
        print("Splitting data and training model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use a smaller parameter grid for faster training if dataset is large
        if X_train.shape[0] > 10000:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        print("Performing hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        # Use 3-fold CV for faster training
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance analysis
        self.analyze_feature_importance()
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def analyze_feature_importance(self, top_n: int = 20):
        """Analyze and display feature importance."""
        if self.model is None or self.feature_names is None:
            print("Model not trained yet!")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(feature_importance.head(top_n))
        
        # Save feature importance plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.title('Top Feature Importance - Disease Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance
    
    def save_model(self):
        """Save the trained model and preprocessors."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save model
        joblib.dump(self.model, 'models/ml_model.pkl')
        print("Saved model to models/ml_model.pkl")
        
        # Save vectorizer
        save_preprocessor(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        print("Saved TF-IDF vectorizer to models/tfidf_vectorizer.pkl")
        
        # Save label encoder
        save_preprocessor(self.label_encoder, 'models/label_encoder.pkl')
        print("Saved label encoder to models/label_encoder.pkl")
        
        # Save feature names for analysis
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print("Saved feature names to models/feature_names.pkl")
    
    def cross_validate(self, X, y, cv: int = 5):
        """Perform cross-validation to assess model stability."""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores

def main():
    """Main training pipeline."""
    print("Starting ML Disease Prediction Model Training...")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MLDiseasePredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_data()
        X, y, cleaned_symptoms, diseases = predictor.preprocess_data(df)
        
        # Train model
        X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
        
        # Cross-validation
        predictor.cross_validate(X, y)
        
        # Save model and preprocessors
        predictor.save_model()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("Model and preprocessors saved to models/ directory")
        
        # Test with sample predictions
        print("\nTesting with sample symptoms:")
        sample_symptoms = [
            "high fever and body pain",
            "cough and cold symptoms",
            "stomach ache and nausea",
            "chest pain and breathing difficulty"
        ]
        
        for symptom in sample_symptoms:
            cleaned = predictor.preprocessor.clean_text(symptom)
            if cleaned.strip():
                X_sample = predictor.vectorizer.transform([cleaned])
                prediction = predictor.model.predict(X_sample)[0]
                disease = predictor.label_encoder.inverse_transform([prediction])[0]
                probability = predictor.model.predict_proba(X_sample)[0].max()
                
                print(f"Symptom: '{symptom}'")
                print(f"Predicted Disease: {disease} (confidence: {probability:.3f})")
                print("-" * 40)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset file exists at data/symptoms_dataset.csv")
        print("Dataset format should be: symptoms,disease")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
