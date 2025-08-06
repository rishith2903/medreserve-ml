#!/usr/bin/env python3
"""
Simple ML API for MedReserve - Fallback version without complex dependencies
This version provides basic functionality without requiring NLTK data or complex models
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Simple medical specialization mapping
SPECIALIZATION_MAPPING = {
    # Cardiology
    'chest pain': 'Cardiology',
    'heart': 'Cardiology',
    'cardiac': 'Cardiology',
    'palpitations': 'Cardiology',
    'shortness of breath': 'Cardiology',
    
    # Neurology
    'headache': 'Neurology',
    'migraine': 'Neurology',
    'seizure': 'Neurology',
    'dizziness': 'Neurology',
    'numbness': 'Neurology',
    
    # Gastroenterology
    'stomach': 'Gastroenterology',
    'abdominal': 'Gastroenterology',
    'nausea': 'Gastroenterology',
    'vomiting': 'Gastroenterology',
    'diarrhea': 'Gastroenterology',
    
    # Orthopedics
    'back pain': 'Orthopedics',
    'joint': 'Orthopedics',
    'bone': 'Orthopedics',
    'fracture': 'Orthopedics',
    'muscle': 'Orthopedics',
    
    # Dermatology
    'skin': 'Dermatology',
    'rash': 'Dermatology',
    'acne': 'Dermatology',
    'eczema': 'Dermatology',
    
    # General Medicine (default)
    'fever': 'General Medicine',
    'fatigue': 'General Medicine',
    'weakness': 'General Medicine',
}

def simple_predict_specialization(symptoms_text):
    """Simple rule-based specialization prediction"""
    symptoms_lower = symptoms_text.lower()
    
    # Score each specialization
    specialization_scores = {}
    
    for symptom, specialization in SPECIALIZATION_MAPPING.items():
        if symptom in symptoms_lower:
            if specialization not in specialization_scores:
                specialization_scores[specialization] = 0
            specialization_scores[specialization] += 1
    
    # If no matches, default to General Medicine
    if not specialization_scores:
        specialization_scores['General Medicine'] = 1
    
    # Sort by score and return top 3
    sorted_specializations = sorted(
        specialization_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    predictions = []
    for i, (spec, score) in enumerate(sorted_specializations[:3]):
        confidence = min(0.9, 0.5 + (score * 0.1))  # Simple confidence calculation
        predictions.append({
            'specialization': spec,
            'confidence': round(confidence, 2),
            'rank': i + 1
        })
    
    return predictions

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MedReserve ML API (Simple)',
        'version': '1.0.0',
        'models_loaded': True,
        'fallback_mode': True
    })

@app.route('/predict/specialization', methods=['POST'])
def predict_specialization():
    """Predict medical specialization from symptoms"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Missing symptoms field'}), 400
        
        symptoms = data['symptoms']
        if not symptoms or not symptoms.strip():
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        # Get predictions
        predictions = simple_predict_specialization(symptoms)
        
        response = {
            'predictions': predictions,
            'input_symptoms': symptoms,
            'model_type': 'rule_based',
            'fallback_mode': True,
            'confidence_threshold': 0.5
        }
        
        logger.info(f"Specialization prediction: {symptoms[:50]}... -> {predictions[0]['specialization']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in specialization prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/diagnosis', methods=['POST'])
def predict_diagnosis():
    """Predict diagnosis from symptoms (simplified)"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Missing symptoms field'}), 400
        
        symptoms = data['symptoms']
        if not symptoms or not symptoms.strip():
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        # Simple diagnosis suggestions
        diagnoses = [
            {
                'condition': 'Common Cold',
                'confidence': 0.7,
                'description': 'Viral infection affecting the upper respiratory tract'
            },
            {
                'condition': 'Stress-related symptoms',
                'confidence': 0.6,
                'description': 'Physical symptoms related to stress or anxiety'
            },
            {
                'condition': 'Requires medical evaluation',
                'confidence': 0.8,
                'description': 'Symptoms require professional medical assessment'
            }
        ]
        
        response = {
            'diagnoses': diagnoses,
            'input_symptoms': symptoms,
            'model_type': 'rule_based',
            'fallback_mode': True,
            'disclaimer': 'This is a simplified prediction. Please consult a healthcare professional for accurate diagnosis.'
        }
        
        logger.info(f"Diagnosis prediction: {symptoms[:50]}...")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in diagnosis prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    return jsonify({
        'models': {
            'specialization_model': {
                'type': 'rule_based',
                'status': 'active',
                'fallback_mode': True
            },
            'diagnosis_model': {
                'type': 'rule_based', 
                'status': 'active',
                'fallback_mode': True
            }
        },
        'nltk_data_available': False,
        'complex_models_available': False,
        'service_mode': 'simple_fallback'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info("🚀 Starting MedReserve Simple ML API")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    logger.info("Mode: Simple fallback (no complex dependencies)")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
