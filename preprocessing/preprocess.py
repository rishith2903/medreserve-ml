"""
Text preprocessing module for disease prediction pipeline.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import joblib
import pickle

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

class TextPreprocessor:
    """Text preprocessing utilities for disease prediction."""
    
    def __init__(self):
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        ])
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text using spaCy or fallback methods.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and lemmatized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (but keep medical terms like "fever 102")
        text = re.sub(r'\b\d+\b', '', text)
        
        if nlp is not None:
            # Use spaCy for advanced preprocessing
            doc = nlp(text)
            tokens = []
            
            for token in doc:
                # Skip punctuation, spaces, and stop words
                if (not token.is_punct and 
                    not token.is_space and 
                    not token.is_stop and
                    len(token.text) > 2 and
                    token.text not in self.stop_words):
                    # Use lemma for better normalization
                    tokens.append(token.lemma_)
            
            return ' '.join(tokens)
        else:
            # Fallback preprocessing without spaCy
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Tokenize and remove stop words
            tokens = [word for word in text.split() 
                     if word not in self.stop_words and len(word) > 2]
            
            return ' '.join(tokens)
    
    def get_tfidf_features(self, corpus: List[str], max_features: int = 5000) -> Tuple[Any, TfidfVectorizer]:
        """
        Convert text corpus to TF-IDF features.
        
        Args:
            corpus (List[str]): List of cleaned text documents
            max_features (int): Maximum number of features
            
        Returns:
            Tuple[Any, TfidfVectorizer]: TF-IDF matrix and fitted vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            stop_words='english'
        )
        
        X = vectorizer.fit_transform(corpus)
        return X, vectorizer
    
    def get_tokenizer_sequences(self, texts: List[str], max_words: int = 10000, 
                               max_len: int = 100) -> Tuple[np.ndarray, Tokenizer]:
        """
        Convert texts to tokenized and padded sequences for deep learning.
        
        Args:
            texts (List[str]): List of cleaned text documents
            max_words (int): Maximum number of words in vocabulary
            max_len (int): Maximum sequence length
            
        Returns:
            Tuple[np.ndarray, Tokenizer]: Padded sequences and fitted tokenizer
        """
        tokenizer = Tokenizer(
            num_words=max_words,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
        
        return padded_sequences, tokenizer

def clean_text(text: str) -> str:
    """Standalone function for text cleaning."""
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)

def get_tfidf_features(corpus: List[str], max_features: int = 5000) -> Tuple[Any, TfidfVectorizer]:
    """Standalone function for TF-IDF feature extraction."""
    preprocessor = TextPreprocessor()
    return preprocessor.get_tfidf_features(corpus, max_features)

def get_tokenizer_sequences(texts: List[str], max_words: int = 10000, 
                           max_len: int = 100) -> Tuple[np.ndarray, Tokenizer]:
    """Standalone function for tokenization and padding."""
    preprocessor = TextPreprocessor()
    return preprocessor.get_tokenizer_sequences(texts, max_words, max_len)

def save_preprocessor(obj: Any, filepath: str) -> None:
    """Save preprocessor object (vectorizer/tokenizer) to file."""
    if filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    else:
        joblib.dump(obj, filepath)

def load_preprocessor(filepath: str) -> Any:
    """Load preprocessor object from file."""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return joblib.load(filepath)

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessing functions
    sample_texts = [
        "I have high fever and body pain with headache",
        "Experiencing severe cough and cold symptoms",
        "Feeling nauseous with stomach ache and diarrhea",
        "Having chest pain and difficulty breathing"
    ]
    
    print("Testing text preprocessing...")
    preprocessor = TextPreprocessor()
    
    for text in sample_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 50)
    
    # Test TF-IDF
    print("\nTesting TF-IDF features...")
    cleaned_texts = [preprocessor.clean_text(text) for text in sample_texts]
    X_tfidf, vectorizer = preprocessor.get_tfidf_features(cleaned_texts)
    print(f"TF-IDF shape: {X_tfidf.shape}")
    print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")
    
    # Test tokenization
    print("\nTesting tokenization...")
    sequences, tokenizer = preprocessor.get_tokenizer_sequences(cleaned_texts, max_len=20)
    print(f"Sequences shape: {sequences.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Sample sequence: {sequences[0]}")
