"""
Deep Learning prediction script for disease prediction.
Uses trained Keras Tokenizer + LSTM model.
"""

import os
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.preprocess import TextPreprocessor, load_preprocessor

class DLDiseasePredictor:
    """Deep Learning-based disease prediction using LSTM."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Load all required models and preprocessors."""
        try:
            # Load DL model
            model_path = os.path.join(self.models_dir, "dl_model.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DL model not found at {model_path}")
            self.model = load_model(model_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.models_dir, "tokenizer.pkl")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            self.tokenizer = load_preprocessor(tokenizer_path)
            
            # Load label encoder
            encoder_path = os.path.join(self.models_dir, "dl_label_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
            self.label_encoder = load_preprocessor(encoder_path)
            
            # Load configuration
            config_path = os.path.join(self.models_dir, "dl_config.pkl")
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    self.config = pickle.load(f)
            else:
                # Default configuration
                self.config = {
                    'max_len': 100,
                    'max_words': 10000,
                    'embedding_dim': 128
                }
            
            self.is_loaded = True
            print("DL models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading DL models: {e}")
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
            
            # Convert to sequences
            sequences = self.tokenizer.texts_to_sequences([cleaned_text])
            X = pad_sequences(sequences, maxlen=self.config['max_len'], padding='post')
            
            # Get prediction probabilities
            probabilities = self.model.predict(X, verbose=0)[0]
            predicted_class = np.argmax(probabilities)
            
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
                "model_type": "deep_learning",
                "sequence_length": len(sequences[0]) if sequences[0] else 0
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
        
        try:
            # Preprocess all texts
            cleaned_texts = [self.preprocessor.clean_text(text) for text in symptom_texts]
            
            # Convert to sequences
            sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            X = pad_sequences(sequences, maxlen=self.config['max_len'], padding='post')
            
            # Get predictions
            probabilities = self.model.predict(X, verbose=0)
            
            results = []
            for i, (original_text, cleaned_text, probs) in enumerate(zip(symptom_texts, cleaned_texts, probabilities)):
                if not cleaned_text.strip():
                    results.append({
                        "error": "No valid symptoms found after preprocessing",
                        "original_text": original_text,
                        "cleaned_text": cleaned_text
                    })
                    continue
                
                predicted_class = np.argmax(probs)
                predicted_disease = self.label_encoder.inverse_transform([predicted_class])[0]
                confidence = probs[predicted_class]
                
                # Get top 3 predictions
                top_indices = np.argsort(probs)[::-1][:3]
                top_predictions = []
                
                for idx in top_indices:
                    disease = self.label_encoder.inverse_transform([idx])[0]
                    prob = probs[idx]
                    top_predictions.append({
                        "disease": disease,
                        "confidence": float(prob)
                    })
                
                results.append({
                    "predicted_disease": predicted_disease,
                    "confidence": float(confidence),
                    "top_predictions": top_predictions,
                    "original_text": original_text,
                    "cleaned_text": cleaned_text,
                    "model_type": "deep_learning",
                    "sequence_length": len(sequences[i]) if sequences[i] else 0
                })
            
            return results

        except Exception as e:
            return [{"error": f"Batch prediction failed: {str(e)}"} for _ in symptom_texts]

    def get_attention_weights(self, symptom_text: str) -> Dict:
        """
        Get attention-like analysis for LSTM predictions.

        Args:
            symptom_text (str): Symptom description

        Returns:
            Dict: Analysis with word importance
        """
        if not self.is_loaded:
            return {"error": "Models not loaded"}

        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(symptom_text)
            if not cleaned_text.strip():
                return {"error": "No valid symptoms found"}

            # Convert to sequence
            sequences = self.tokenizer.texts_to_sequences([cleaned_text])
            X = pad_sequences(sequences, maxlen=self.config['max_len'], padding='post')

            # Get base prediction
            base_pred = self.model.predict(X, verbose=0)[0]
            base_class = np.argmax(base_pred)
            base_confidence = base_pred[base_class]

            # Analyze word importance by masking
            words = cleaned_text.split()
            word_importance = []

            for i, word in enumerate(words):
                # Create masked version
                masked_words = words.copy()
                masked_words[i] = ""  # Remove word
                masked_text = " ".join(masked_words).strip()

                if masked_text:
                    masked_seq = self.tokenizer.texts_to_sequences([masked_text])
                    masked_X = pad_sequences(masked_seq, maxlen=self.config['max_len'], padding='post')
                    masked_pred = self.model.predict(masked_X, verbose=0)[0]
                    masked_confidence = masked_pred[base_class]

                    # Importance = drop in confidence when word is removed
                    importance = base_confidence - masked_confidence
                else:
                    importance = base_confidence

                word_importance.append({
                    "word": word,
                    "importance": float(importance),
                    "position": i
                })

            # Sort by importance
            word_importance.sort(key=lambda x: x['importance'], reverse=True)

            return {
                "original_text": symptom_text,
                "cleaned_text": cleaned_text,
                "predicted_disease": self.label_encoder.inverse_transform([base_class])[0],
                "confidence": float(base_confidence),
                "word_importance": word_importance
            }

        except Exception as e:
            return {"error": f"Attention analysis failed: {str(e)}"}

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"error": "Models not loaded"}

        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))

        return {
            "model_type": "LSTM",
            "architecture": "\n".join(model_summary),
            "config": self.config,
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "vocab_size": len(self.tokenizer.word_index),
            "max_sequence_length": self.config['max_len']
        }

# Standalone prediction functions for easy import
_predictor_instance = None

def get_predictor() -> DLDiseasePredictor:
    """Get or create predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DLDiseasePredictor()
    return _predictor_instance

def predict_with_dl(symptom_text: str) -> Dict:
    """
    Standalone function for DL prediction.

    Args:
        symptom_text (str): Symptom description

    Returns:
        Dict: Prediction result
    """
    predictor = get_predictor()
    return predictor.predict_single(symptom_text)

def predict_batch_dl(symptom_texts: List[str]) -> List[Dict]:
    """
    Standalone function for batch DL prediction.

    Args:
        symptom_texts (List[str]): List of symptom descriptions

    Returns:
        List[Dict]: List of prediction results
    """
    predictor = get_predictor()
    return predictor.predict_batch(symptom_texts)

def analyze_prediction_dl(symptom_text: str) -> Dict:
    """
    Standalone function for prediction analysis.

    Args:
        symptom_text (str): Symptom description

    Returns:
        Dict: Analysis result
    """
    predictor = get_predictor()
    return predictor.get_attention_weights(symptom_text)

# Example usage and testing
if __name__ == "__main__":
    print("Testing DL Disease Prediction...")
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

    predictor = DLDiseasePredictor()

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

        # Test attention analysis
        print("\nWord Importance Analysis Example:")
        analysis = predictor.get_attention_weights(test_symptoms[0])
        if "error" not in analysis:
            print(f"Important words for: {test_symptoms[0]}")
            for word_info in analysis['word_importance'][:5]:
                print(f"  {word_info['word']}: {word_info['importance']:.4f}")
    else:
        print("Failed to load models. Please train the models first.")
        print("Run: python train_dl_model.py")
