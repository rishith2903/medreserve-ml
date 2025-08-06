"""
NLP Pipeline for MedReserve AI
Shared preprocessing pipeline for both patient and doctor models
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Download required NLTK data with error handling
def download_nltk_data():
    """Download NLTK data with proper error handling"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]

    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {download_name}")
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download {download_name}: {e}")
                continue

    # Try punkt_tab for newer NLTK versions, but don't fail if it doesn't exist
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            # punkt_tab doesn't exist in older versions, use punkt instead
            pass

# Download NLTK data
download_nltk_data()

class MedicalNLPPipeline:
    """
    Comprehensive NLP pipeline for medical text processing
    """
    
    def __init__(self):
        # Initialize with fallback support for NLTK data
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: WordNetLemmatizer not available: {e}")
            self.lemmatizer = None

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: NLTK stopwords not available: {e}")
            # Fallback stop words
            self.stop_words = {
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during',
                'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
                'over', 'under', 'again', 'further', 'then', 'once'
            }

        # Medical-specific stop words to remove
        self.medical_stop_words = {
            'patient', 'feel', 'feeling', 'experience', 'experiencing',
            'have', 'has', 'had', 'get', 'getting', 'got', 'seem', 'seems',
            'like', 'also', 'sometimes', 'often', 'usually', 'always',
            'doctor', 'physician', 'medical', 'health', 'healthcare'
        }

        self.stop_words.update(self.medical_stop_words)
        
        # Medical symptom keywords to preserve
        self.medical_keywords = {
            'pain', 'ache', 'fever', 'headache', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'fatigue', 'weakness', 'dizziness',
            'shortness', 'breath', 'cough', 'chest', 'abdominal', 'back',
            'joint', 'muscle', 'skin', 'rash', 'swelling', 'inflammation',
            'bleeding', 'discharge', 'infection', 'allergy', 'asthma',
            'diabetes', 'hypertension', 'migraine', 'anxiety', 'depression'
        }
        
        self.tfidf_vectorizer = None
        self.label_encoder = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize medical text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers unless they're part of medical terms
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize medical text with fallback support
        """
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK tokenizer not available: {e}")
            # Simple fallback tokenization
            tokens = text.split()
        
        # Remove stop words but preserve medical keywords
        tokens = [
            token for token in tokens 
            if token not in self.stop_words or token in self.medical_keywords
        ]
        
        # Lemmatize with fallback
        if self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except Exception as e:
                print(f"Warning: Lemmatization failed: {e}")
                # Continue without lemmatization
                pass
        
        # Remove very short tokens (less than 2 characters)
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def preprocess_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Complete preprocessing pipeline for medical text
        """
        if isinstance(text, list):
            return [self._preprocess_single_text(t) for t in text]
        else:
            return self._preprocess_single_text(text)
    
    def _preprocess_single_text(self, text: str) -> str:
        """
        Preprocess a single text string
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        # Join back to string
        return ' '.join(tokens)
    
    def fit_vectorizer(self, texts: List[str], max_features: int = 5000):
        """
        Fit TF-IDF vectorizer on training texts
        """
        # Preprocess texts
        processed_texts = self.preprocess_text(texts)
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams for better medical context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        self.tfidf_vectorizer.fit(processed_texts)
        
        return self.tfidf_vectorizer
    
    def vectorize_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to TF-IDF vectors
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_vectorizer first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = self.preprocess_text(texts)
        
        # Transform to vectors
        vectors = self.tfidf_vectorizer.transform(processed_texts)
        
        return vectors.toarray()
    
    def fit_label_encoder(self, labels: List[str]):
        """
        Fit label encoder for target labels
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        return self.label_encoder
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode labels to numerical format
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit_label_encoder first.")
        
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode numerical labels back to text
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def save_pipeline(self, filepath: str):
        """
        Save the fitted pipeline components
        """
        pipeline_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'stop_words': self.stop_words,
            'medical_keywords': self.medical_keywords
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
    
    def load_pipeline(self, filepath: str):
        """
        Load a saved pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.tfidf_vectorizer = pipeline_data['tfidf_vectorizer']
        self.label_encoder = pipeline_data['label_encoder']
        self.stop_words = pipeline_data.get('stop_words', self.stop_words)
        self.medical_keywords = pipeline_data.get('medical_keywords', self.medical_keywords)

def create_symptom_combinations(symptoms: List[str]) -> List[str]:
    """
    Create meaningful combinations of symptoms for better prediction
    """
    combinations = symptoms.copy()
    
    # Add common medical combinations
    symptom_text = ' '.join(symptoms).lower()
    
    if 'fever' in symptom_text and 'headache' in symptom_text:
        combinations.append('fever headache')
    
    if 'chest' in symptom_text and 'pain' in symptom_text:
        combinations.append('chest pain')
    
    if 'shortness' in symptom_text and 'breath' in symptom_text:
        combinations.append('shortness of breath')
    
    if 'abdominal' in symptom_text and 'pain' in symptom_text:
        combinations.append('abdominal pain')
    
    return combinations

# Example usage and testing
if __name__ == "__main__":
    # Test the NLP pipeline
    nlp = MedicalNLPPipeline()
    
    # Test text preprocessing
    test_texts = [
        "I have severe headache and fever for 3 days",
        "Patient experiencing chest pain and shortness of breath",
        "Feeling nauseous with abdominal pain and diarrhea"
    ]
    
    print("Original texts:")
    for text in test_texts:
        print(f"- {text}")
    
    print("\nProcessed texts:")
    processed = nlp.preprocess_text(test_texts)
    for text in processed:
        print(f"- {text}")
    
    # Test vectorization
    nlp.fit_vectorizer(test_texts, max_features=100)
    vectors = nlp.vectorize_text(test_texts)
    print(f"\nVector shape: {vectors.shape}")
    print(f"Feature names (first 10): {nlp.tfidf_vectorizer.get_feature_names_out()[:10]}")
