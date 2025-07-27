# Disease Prediction Pipeline

A complete machine learning and deep learning-based disease prediction system using Natural Language Processing (NLP) to analyze symptom descriptions and predict the most likely diseases.

## 🎯 Overview

This pipeline takes user symptom descriptions like "I have high fever and body pain" and predicts the most likely disease using two complementary approaches:

1. **Machine Learning Model**: TF-IDF features + RandomForestClassifier
2. **Deep Learning Model**: Keras Tokenizer + LSTM Neural Network

## 📁 Project Structure

```
backend/ml/
├── data/
│   ├── symptoms_dataset.csv           # Main dataset (place your data here)
│   └── sample_symptoms_dataset.csv    # Sample dataset for testing
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py                  # Text preprocessing utilities
├── models/
│   ├── ml_model.pkl                   # Trained RandomForest model
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── dl_model.h5                    # Trained LSTM model
│   ├── tokenizer.pkl                  # Keras tokenizer
│   ├── label_encoder.pkl              # Label encoder for ML
│   ├── dl_label_encoder.pkl           # Label encoder for DL
│   └── dl_config.pkl                  # DL model configuration
├── train_ml_model.py                  # Train ML model
├── train_dl_model.py                  # Train DL model
├── predict_ml.py                      # ML prediction functions
├── predict_dl.py                      # DL prediction functions
└── requirements.txt                   # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Prepare Dataset

Place your dataset as `data/symptoms_dataset.csv` with the following format:

```csv
symptoms,disease
"high fever body ache headache chills fatigue",flu
"cough cold runny nose sore throat sneezing",common_cold
"stomach ache nausea vomiting diarrhea",gastroenteritis
```

**Note**: A sample dataset is provided at `data/sample_symptoms_dataset.csv` for testing.

### 3. Train Models

```bash
# Train Machine Learning model
python train_ml_model.py

# Train Deep Learning model
python train_dl_model.py
```

### 4. Make Predictions

```python
# Using ML model
from predict_ml import predict_with_ml

result = predict_with_ml("I have high fever and body pain")
print(f"Predicted disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")

# Using DL model
from predict_dl import predict_with_dl

result = predict_with_dl("I have high fever and body pain")
print(f"Predicted disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 🔧 Features

### Text Preprocessing
- **spaCy-based NLP**: Advanced tokenization, lemmatization, and stop word removal
- **Fallback processing**: Works even without spaCy installation
- **Medical text optimization**: Handles medical terminology and symptoms

### Machine Learning Model
- **TF-IDF Vectorization**: Converts text to numerical features
- **RandomForest Classifier**: Robust ensemble method
- **Hyperparameter Tuning**: Automated grid search for optimal parameters
- **Feature Importance**: Analysis of most important symptom keywords

### Deep Learning Model
- **LSTM Architecture**: Bidirectional LSTM for sequence modeling
- **Embedding Layer**: Dense vector representations of words
- **Attention Analysis**: Word importance analysis for predictions
- **Early Stopping**: Prevents overfitting during training

### Prediction Features
- **Single & Batch Prediction**: Process one or multiple symptom descriptions
- **Confidence Scores**: Probability estimates for predictions
- **Top-N Predictions**: Multiple disease possibilities ranked by confidence
- **Feature Analysis**: Understanding which symptoms drive predictions

## 📊 Model Performance

### Machine Learning Model
- **Algorithm**: RandomForestClassifier with TF-IDF features
- **Features**: Up to 5000 TF-IDF features (unigrams + bigrams)
- **Evaluation**: Cross-validation, accuracy, F1-score, confusion matrix
- **Interpretability**: Feature importance analysis

### Deep Learning Model
- **Architecture**: Bidirectional LSTM with embedding layer
- **Sequence Length**: Up to 100 tokens per symptom description
- **Vocabulary**: Up to 10,000 most frequent words
- **Regularization**: Dropout layers and early stopping

## 🔌 Integration with FastAPI/Flask

### FastAPI Integration

```python
from fastapi import FastAPI
from predict_ml import predict_with_ml
from predict_dl import predict_with_dl

app = FastAPI()

@app.post("/predict/ml")
def predict_ml_endpoint(symptom_text: str):
    return predict_with_ml(symptom_text)

@app.post("/predict/dl")
def predict_dl_endpoint(symptom_text: str):
    return predict_with_dl(symptom_text)

@app.post("/predict/ensemble")
def predict_ensemble(symptom_text: str):
    ml_result = predict_with_ml(symptom_text)
    dl_result = predict_with_dl(symptom_text)
    
    return {
        "ml_prediction": ml_result,
        "dl_prediction": dl_result,
        "ensemble_confidence": (ml_result['confidence'] + dl_result['confidence']) / 2
    }
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from predict_ml import predict_with_ml
from predict_dl import predict_with_dl

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptom_text = data.get('symptoms', '')
    
    ml_result = predict_with_ml(symptom_text)
    dl_result = predict_with_dl(symptom_text)
    
    return jsonify({
        'ml_prediction': ml_result,
        'dl_prediction': dl_result
    })
```

## 🧪 Testing

### Test Individual Components

```bash
# Test preprocessing
python preprocessing/preprocess.py

# Test ML prediction
python predict_ml.py

# Test DL prediction
python predict_dl.py
```

### Example API Calls

```python
# Test with sample symptoms
test_symptoms = [
    "I have high fever and body pain with headache",
    "Experiencing severe cough and cold symptoms",
    "Feeling nauseous with stomach ache and diarrhea"
]

from predict_ml import predict_batch_ml
from predict_dl import predict_batch_dl

ml_results = predict_batch_ml(test_symptoms)
dl_results = predict_batch_dl(test_symptoms)
```

## 📈 Performance Optimization

### For Production Use
1. **Model Caching**: Load models once and reuse
2. **Batch Processing**: Process multiple predictions together
3. **GPU Support**: Use TensorFlow-GPU for faster DL inference
4. **Model Quantization**: Reduce model size for deployment

### Memory Management
- Models are loaded lazily (only when first prediction is made)
- Singleton pattern ensures single model instance per process
- Efficient preprocessing with spaCy pipeline caching

## 🔍 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **TensorFlow Installation Issues**
   ```bash
   pip install tensorflow==2.10.0
   ```

3. **Memory Issues with Large Datasets**
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller batch sizes for training
   - Consider data sampling for very large datasets

4. **Low Prediction Accuracy**
   - Ensure dataset quality and balance
   - Increase training data size
   - Tune hyperparameters
   - Check for data leakage

## 📝 Dataset Requirements

### Format
- **CSV file** with columns: `symptoms`, `disease`
- **Text encoding**: UTF-8
- **Minimum samples**: 100+ per disease class for good performance

### Quality Guidelines
- Clear, descriptive symptom descriptions
- Consistent disease naming
- Balanced distribution across disease classes
- Medical accuracy in symptom-disease mappings

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
CMD ["python", "app.py"]
```

### Cloud Deployment
- **AWS**: Use SageMaker for model hosting
- **Google Cloud**: Deploy on AI Platform
- **Azure**: Use Azure ML for model deployment

## 📄 License

This project is part of the MedReserve AI healthcare management system.

## 🤝 Contributing

1. Ensure code follows PEP 8 standards
2. Add unit tests for new features
3. Update documentation for API changes
4. Test with sample dataset before submitting
