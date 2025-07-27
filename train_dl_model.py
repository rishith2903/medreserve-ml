"""
Deep Learning model training script for disease prediction.
Uses Keras Tokenizer + LSTM neural network.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.preprocess import TextPreprocessor, save_preprocessor
import pickle

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DLDiseasePredictor:
    """Deep Learning based disease prediction using LSTM."""
    
    def __init__(self, data_path: str = "data/symptoms_dataset.csv"):
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 100
        self.max_words = 10000
        self.embedding_dim = 128
        self.history = None
        
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
        print("Preprocessing text data for deep learning...")
        
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
        
        # Convert to tokenized sequences
        X, self.tokenizer = self.preprocessor.get_tokenizer_sequences(
            cleaned_symptoms, max_words=self.max_words, max_len=self.max_len
        )
        
        # Encode disease labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(diseases)
        y = to_categorical(y_encoded)
        
        print(f"Sequence matrix shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Max sequence length: {self.max_len}")
        
        return X, y, cleaned_symptoms, diseases
    
    def build_model(self, num_classes: int, vocab_size: int) -> Sequential:
        """Build the LSTM model architecture."""
        print("Building LSTM model...")
        
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True
            ),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X, y, test_size: float = 0.2, validation_split: float = 0.1, 
                   epochs: int = 50, batch_size: int = 32, random_state: int = 42):
        """Train the LSTM model."""
        print("Training deep learning model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y.argmax(axis=1)
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Build model
        vocab_size = min(self.max_words, len(self.tokenizer.word_index)) + 1
        num_classes = y.shape[1]
        self.model = self.build_model(num_classes, vocab_size)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_dl_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot training history
        self.plot_training_history()
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history plot saved to models/training_history.png")
    
    def save_model(self):
        """Save the trained model and preprocessors."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save model
        self.model.save('models/dl_model.h5')
        print("Saved model to models/dl_model.h5")
        
        # Save tokenizer
        save_preprocessor(self.tokenizer, 'models/tokenizer.pkl')
        print("Saved tokenizer to models/tokenizer.pkl")
        
        # Save label encoder
        save_preprocessor(self.label_encoder, 'models/dl_label_encoder.pkl')
        print("Saved label encoder to models/dl_label_encoder.pkl")
        
        # Save model configuration
        config = {
            'max_len': self.max_len,
            'max_words': self.max_words,
            'embedding_dim': self.embedding_dim,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'num_classes': len(self.label_encoder.classes_)
        }
        
        with open('models/dl_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        print("Saved model configuration to models/dl_config.pkl")

def main():
    """Main training pipeline."""
    print("Starting Deep Learning Disease Prediction Model Training...")
    print("=" * 70)
    
    # Initialize predictor
    predictor = DLDiseasePredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_data()
        X, y, cleaned_symptoms, diseases = predictor.preprocess_data(df)
        
        # Train model
        X_train, X_test, y_train, y_test, y_pred = predictor.train_model(
            X, y, epochs=30, batch_size=32
        )
        
        # Save model and preprocessors
        predictor.save_model()
        
        print("\n" + "=" * 70)
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
                # Convert to sequence
                sequences = predictor.tokenizer.texts_to_sequences([cleaned])
                X_sample = tf.keras.preprocessing.sequence.pad_sequences(
                    sequences, maxlen=predictor.max_len, padding='post'
                )
                
                # Predict
                prediction = predictor.model.predict(X_sample, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
                disease = predictor.label_encoder.inverse_transform([predicted_class])[0]
                
                print(f"Symptom: '{symptom}'")
                print(f"Predicted Disease: {disease} (confidence: {confidence:.3f})")
                print("-" * 50)
        
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
