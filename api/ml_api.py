"""
Flask API for MedReserve AI ML Models
Provides REST endpoints for patient specialization and doctor diagnosis predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict.predict_specialization import SpecializationPredictor, fallback_specialization_prediction
from predict.predict_disease_medicine import DiseaseMedicinePredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictors
specialization_predictor = None
diagnosis_predictor = None

def initialize_models():
    """
    Initialize ML models on startup
    """
    global specialization_predictor, diagnosis_predictor
    
    try:
        # Initialize specialization predictor
        specialization_predictor = SpecializationPredictor()
        try:
            specialization_predictor.load_model()
            logger.info("Specialization model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load specialization model: {e}")
            specialization_predictor = None
        
        # Initialize diagnosis predictor
        diagnosis_predictor = DiseaseMedicinePredictor()
        try:
            diagnosis_predictor.load_models()
            logger.info("Diagnosis models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load diagnosis models: {e}")
            diagnosis_predictor = None
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'specialization_loaded': specialization_predictor is not None and specialization_predictor.is_loaded,
            'diagnosis_loaded': diagnosis_predictor is not None and diagnosis_predictor.is_loaded
        }
    })

@app.route('/predict/specialization', methods=['POST'])
def predict_specialization():
    """
    Predict doctor specializations based on patient symptoms
    
    Expected JSON payload:
    {
        "symptoms": "patient symptom description",
        "top_k": 3  # optional, default 3
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'symptoms' not in data:
            return jsonify({'error': 'Missing required field: symptoms'}), 400
        
        symptoms = data['symptoms']
        top_k = data.get('top_k', 3)
        
        # Validate inputs
        if not symptoms or not symptoms.strip():
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            return jsonify({'error': 'top_k must be an integer between 1 and 10'}), 400
        
        # Make prediction
        if specialization_predictor and specialization_predictor.is_loaded:
            result = specialization_predictor.predict_specializations(symptoms, top_k)
        else:
            # Use fallback prediction
            result = fallback_specialization_prediction(symptoms, top_k)
            result['fallback'] = True
        
        # Add request metadata
        result['request_id'] = f"spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result['input_symptoms'] = symptoms
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_specialization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/diagnosis', methods=['POST'])
def predict_diagnosis():
    """
    Predict diseases and medicines based on doctor-entered symptoms
    
    Expected JSON payload:
    {
        "symptoms": "doctor symptom description",
        "top_diseases": 5,  # optional, default 5
        "top_medicines": 5  # optional, default 5
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'symptoms' not in data:
            return jsonify({'error': 'Missing required field: symptoms'}), 400
        
        symptoms = data['symptoms']
        top_diseases = data.get('top_diseases', 5)
        top_medicines = data.get('top_medicines', 5)
        
        # Validate inputs
        if not symptoms or not symptoms.strip():
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        if not isinstance(top_diseases, int) or top_diseases < 1 or top_diseases > 20:
            return jsonify({'error': 'top_diseases must be an integer between 1 and 20'}), 400
        
        if not isinstance(top_medicines, int) or top_medicines < 1 or top_medicines > 20:
            return jsonify({'error': 'top_medicines must be an integer between 1 and 20'}), 400
        
        # Make prediction
        if diagnosis_predictor:
            result = diagnosis_predictor.predict_diagnosis(symptoms, top_diseases, top_medicines)
        else:
            # Initialize new predictor if needed
            temp_predictor = DiseaseMedicinePredictor()
            result = temp_predictor.predict_diagnosis(symptoms, top_diseases, top_medicines)
            result['fallback'] = True
        
        # Add request metadata
        result['request_id'] = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result['input_symptoms'] = symptoms
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_diagnosis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/models/info', methods=['GET'])
def get_models_info():
    """
    Get information about loaded models
    """
    try:
        info = {
            'specialization_model': None,
            'diagnosis_model': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if specialization_predictor:
            info['specialization_model'] = specialization_predictor.get_model_info()
        
        if diagnosis_predictor:
            info['diagnosis_model'] = diagnosis_predictor.get_model_info()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error in get_models_info: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/batch/specialization', methods=['POST'])
def batch_predict_specialization():
    """
    Batch prediction for multiple symptom descriptions
    
    Expected JSON payload:
    {
        "symptoms_list": ["symptom1", "symptom2", ...],
        "top_k": 3  # optional, default 3
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'symptoms_list' not in data:
            return jsonify({'error': 'Missing required field: symptoms_list'}), 400
        
        symptoms_list = data['symptoms_list']
        top_k = data.get('top_k', 3)
        
        if not isinstance(symptoms_list, list) or len(symptoms_list) == 0:
            return jsonify({'error': 'symptoms_list must be a non-empty list'}), 400
        
        if len(symptoms_list) > 50:
            return jsonify({'error': 'Maximum 50 symptoms allowed per batch'}), 400
        
        # Make batch predictions
        results = []
        for symptoms in symptoms_list:
            if specialization_predictor and specialization_predictor.is_loaded:
                result = specialization_predictor.predict_specializations(symptoms, top_k)
            else:
                result = fallback_specialization_prediction(symptoms, top_k)
                result['fallback'] = True
            results.append(result)
        
        return jsonify({
            'results': results,
            'count': len(results),
            'request_id': f"batch_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })
        
    except Exception as e:
        logger.error(f"Error in batch_predict_specialization: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/batch/diagnosis', methods=['POST'])
def batch_predict_diagnosis():
    """
    Batch prediction for multiple diagnosis requests
    
    Expected JSON payload:
    {
        "symptoms_list": ["symptom1", "symptom2", ...],
        "top_diseases": 5,  # optional, default 5
        "top_medicines": 5  # optional, default 5
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'symptoms_list' not in data:
            return jsonify({'error': 'Missing required field: symptoms_list'}), 400
        
        symptoms_list = data['symptoms_list']
        top_diseases = data.get('top_diseases', 5)
        top_medicines = data.get('top_medicines', 5)
        
        if not isinstance(symptoms_list, list) or len(symptoms_list) == 0:
            return jsonify({'error': 'symptoms_list must be a non-empty list'}), 400
        
        if len(symptoms_list) > 20:
            return jsonify({'error': 'Maximum 20 symptoms allowed per batch'}), 400
        
        # Make batch predictions
        if diagnosis_predictor:
            results = diagnosis_predictor.batch_predict(symptoms_list, top_diseases, top_medicines)
        else:
            temp_predictor = DiseaseMedicinePredictor()
            results = temp_predictor.batch_predict(symptoms_list, top_diseases, top_medicines)
            for result in results:
                result['fallback'] = True
        
        return jsonify({
            'results': results,
            'count': len(results),
            'request_id': f"batch_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })
        
    except Exception as e:
        logger.error(f"Error in batch_predict_diagnosis: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting MedReserve AI ML API on port {port}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict/specialization - Patient to specialization prediction")
    print("  POST /predict/diagnosis - Doctor to diagnosis prediction")
    print("  GET  /models/info - Model information")
    print("  POST /predict/batch/specialization - Batch specialization prediction")
    print("  POST /predict/batch/diagnosis - Batch diagnosis prediction")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
