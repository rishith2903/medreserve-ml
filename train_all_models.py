#!/usr/bin/env python3
"""
MedReserve AI - Complete Model Training Script
Trains both patient-to-specialization and doctor-to-diagnosis models
"""

import os
import sys
import time
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Setup the training environment
    """
    logger.info("Setting up training environment...")
    
    # Create necessary directories
    directories = [
        'models',
        'logs',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Check if datasets exist
    dataset_paths = [
        'dataset/Disease-Symptom Dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv',
        'dataset/Doctor\'s Specialty Recommendation/Doctor_Versus_Disease.csv',
        'dataset/Symptom2Disease/Symptom2Disease.csv',
        'dataset/Disease Symptoms and Patient Profile Dataset/Disease_symptom_and_patient_profile_dataset.csv'
    ]
    
    missing_datasets = []
    for path in dataset_paths:
        if not os.path.exists(path):
            missing_datasets.append(path)
    
    if missing_datasets:
        logger.warning("Missing datasets:")
        for dataset in missing_datasets:
            logger.warning(f"  - {dataset}")
        logger.warning("Training will proceed with available datasets")
    else:
        logger.info("All datasets found!")
    
    return True

def install_dependencies():
    """
    Install required dependencies
    """
    logger.info("Installing dependencies...")
    
    try:
        import subprocess
        
        # Install basic requirements
        basic_requirements = [
            'numpy', 'pandas', 'scikit-learn', 'nltk', 
            'flask', 'flask-cors', 'joblib'
        ]
        
        for package in basic_requirements:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package} already installed")
            except ImportError:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✓ {package} installed successfully")
        
        # Download NLTK data
        import nltk
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
                logger.info(f"✓ NLTK {item} already downloaded")
            except LookupError:
                logger.info(f"Downloading NLTK {item}...")
                nltk.download(item, quiet=True)
                logger.info(f"✓ NLTK {item} downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def train_patient_model():
    """
    Train the patient to specialization model
    """
    logger.info("=" * 60)
    logger.info("TRAINING PATIENT TO SPECIALIZATION MODEL")
    logger.info("=" * 60)
    
    try:
        from train.train_patient_model import PatientSpecializationModel
        
        # Initialize and train model
        model = PatientSpecializationModel()
        
        # Load and prepare data
        logger.info("Loading and preparing patient data...")
        symptoms_df, specializations_df = model.load_and_prepare_data()
        
        # Train model
        logger.info("Training patient model...")
        start_time = time.time()
        model.train_model(symptoms_df, specializations_df)
        training_time = time.time() - start_time
        
        # Save model
        logger.info("Saving patient model...")
        model.save_model()
        
        logger.info(f"✓ Patient model training completed in {training_time:.2f} seconds")
        
        # Test model
        logger.info("Testing patient model...")
        test_symptoms = [
            "I have severe chest pain and shortness of breath",
            "Experiencing headache, dizziness and nausea for 3 days",
            "Skin rash with itching and redness",
            "Joint pain and stiffness in the morning"
        ]
        
        for symptoms in test_symptoms:
            predictions = model.predict_specialization(symptoms, top_k=2)
            logger.info(f"Symptoms: {symptoms}")
            for spec, prob in predictions:
                logger.info(f"  → {spec}: {prob:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training patient model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def train_doctor_model():
    """
    Train the doctor to diagnosis model
    """
    logger.info("=" * 60)
    logger.info("TRAINING DOCTOR TO DIAGNOSIS MODEL")
    logger.info("=" * 60)
    
    try:
        from train.train_doctor_model import DoctorDiagnosisModel
        
        # Initialize and train model
        model = DoctorDiagnosisModel()
        
        # Load and prepare data
        logger.info("Loading and preparing doctor data...")
        symptoms_df, diseases_df, medicines_df = model.load_and_prepare_data()
        
        # Train models
        logger.info("Training doctor models...")
        start_time = time.time()
        model.train_models(symptoms_df, diseases_df, medicines_df)
        training_time = time.time() - start_time
        
        # Save models
        logger.info("Saving doctor models...")
        model.save_models()
        
        logger.info(f"✓ Doctor model training completed in {training_time:.2f} seconds")
        
        # Test model
        logger.info("Testing doctor model...")
        test_symptoms = [
            "Patient presents with chest pain, shortness of breath, and sweating",
            "Severe headache with nausea, vomiting, and sensitivity to light",
            "Persistent cough with fever and difficulty breathing"
        ]
        
        for symptoms in test_symptoms:
            diagnosis = model.predict_diagnosis(symptoms, top_diseases=2, top_medicines=2)
            logger.info(f"Symptoms: {symptoms}")
            logger.info("Diseases:")
            for disease, prob in diagnosis['diseases']:
                logger.info(f"  → {disease}: {prob:.3f}")
            logger.info("Medicines:")
            for medicine, prob in diagnosis['medicines']:
                logger.info(f"  → {medicine}: {prob:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training doctor model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_api():
    """
    Test the ML API
    """
    logger.info("=" * 60)
    logger.info("TESTING ML API")
    logger.info("=" * 60)
    
    try:
        # Test prediction scripts
        from predict.predict_specialization import SpecializationPredictor, fallback_specialization_prediction
        from predict.predict_disease_medicine import DiseaseMedicinePredictor
        
        # Test specialization prediction
        logger.info("Testing specialization prediction...")
        try:
            predictor = SpecializationPredictor()
            predictor.load_model()
            result = predictor.predict_specializations("chest pain and shortness of breath", top_k=2)
            logger.info(f"✓ Specialization prediction working: {len(result['specializations'])} results")
        except Exception as e:
            logger.warning(f"Specialization model not available, testing fallback: {e}")
            result = fallback_specialization_prediction("chest pain and shortness of breath", top_k=2)
            logger.info(f"✓ Fallback specialization prediction working: {len(result['specializations'])} results")
        
        # Test diagnosis prediction
        logger.info("Testing diagnosis prediction...")
        try:
            predictor = DiseaseMedicinePredictor()
            predictor.load_models()
            result = predictor.predict_diagnosis("chest pain and shortness of breath", top_diseases=2, top_medicines=2)
            logger.info(f"✓ Diagnosis prediction working: {len(result['diseases'])} diseases, {len(result['medicines'])} medicines")
        except Exception as e:
            logger.warning(f"Diagnosis model not available, testing fallback: {e}")
            predictor = DiseaseMedicinePredictor()
            result = predictor.predict_diagnosis("chest pain and shortness of breath", top_diseases=2, top_medicines=2)
            logger.info(f"✓ Fallback diagnosis prediction working: {len(result['diseases'])} diseases, {len(result['medicines'])} medicines")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing API: {e}")
        return False

def generate_report():
    """
    Generate training report
    """
    logger.info("=" * 60)
    logger.info("GENERATING TRAINING REPORT")
    logger.info("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': [],
        'files_created': [],
        'status': 'completed'
    }
    
    # Check created files
    model_files = [
        'models/patient_to_specialization_model.pkl',
        'models/patient_nlp_pipeline.pkl',
        'models/patient_label_encoder.pkl',
        'models/doctor_disease_model.pkl',
        'models/doctor_medicine_model.pkl',
        'models/doctor_nlp_pipeline.pkl',
        'models/doctor_disease_encoder.pkl',
        'models/doctor_medicine_encoder.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            report['files_created'].append({
                'file': file_path,
                'size_mb': round(file_size, 2)
            })
            logger.info(f"✓ {file_path} ({file_size:.2f} MB)")
    
    # Save report
    import json
    with open('results/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to: results/training_report.json")
    return report

def main():
    """
    Main training function
    """
    start_time = time.time()
    
    logger.info("🚀 Starting MedReserve AI Model Training")
    logger.info(f"Start time: {datetime.now()}")
    
    # Setup environment
    if not setup_environment():
        logger.error("Failed to setup environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Train models
    patient_success = train_patient_model()
    doctor_success = train_doctor_model()
    
    # Test API
    api_success = test_api()
    
    # Generate report
    report = generate_report()
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Patient Model: {'✓ SUCCESS' if patient_success else '✗ FAILED'}")
    logger.info(f"Doctor Model: {'✓ SUCCESS' if doctor_success else '✗ FAILED'}")
    logger.info(f"API Test: {'✓ SUCCESS' if api_success else '✗ FAILED'}")
    logger.info(f"Total Time: {total_time:.2f} seconds")
    logger.info(f"Files Created: {len(report['files_created'])}")
    
    if patient_success and doctor_success and api_success:
        logger.info("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("Ready to start ML API server with: python api/ml_api.py")
        return True
    else:
        logger.error("❌ SOME MODELS FAILED TO TRAIN")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
