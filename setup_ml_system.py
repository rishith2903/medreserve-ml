#!/usr/bin/env python3
"""
MedReserve AI ML System Setup Script
Complete setup and validation of the dual-model medical AI system
"""

import os
import sys
import subprocess
import time
import json
import requests
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print setup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧠 MedReserve AI ML System                ║
    ║                     Setup & Validation                       ║
    ║                                                              ║
    ║  🎯 Dual-Model Medical AI System                            ║
    ║  📊 Patient → Doctor Specialization                         ║
    ║  🔬 Doctor → Disease & Medicine Prediction                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check Python version compatibility"""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        logger.error(f"Python 3.7+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing dependencies...")
    
    try:
        # Install basic requirements
        basic_packages = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'nltk>=3.6.0',
            'flask>=2.0.0',
            'flask-cors>=3.0.0',
            'joblib>=1.1.0'
        ]
        
        for package in basic_packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info("✓ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for item in nltk_downloads:
            logger.info(f"Downloading NLTK {item}...")
            nltk.download(item, quiet=True)
        
        logger.info("✓ NLTK data downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

def check_datasets():
    """Check if required datasets are available"""
    logger.info("Checking datasets...")
    
    dataset_paths = [
        'dataset/Disease-Symptom Dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv',
        'dataset/Doctor\'s Specialty Recommendation/Doctor_Versus_Disease.csv',
        'dataset/Symptom2Disease/Symptom2Disease.csv',
        'dataset/Disease Symptoms and Patient Profile Dataset/Disease_symptom_and_patient_profile_dataset.csv'
    ]
    
    available_datasets = 0
    for path in dataset_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"✓ Found: {path} ({size_mb:.1f} MB)")
            available_datasets += 1
        else:
            logger.warning(f"✗ Missing: {path}")
    
    if available_datasets > 0:
        logger.info(f"✓ {available_datasets}/{len(dataset_paths)} datasets available")
        return True
    else:
        logger.warning("No datasets found - will use fallback predictions")
        return False

def train_models():
    """Train the ML models"""
    logger.info("Training ML models...")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'train_all_models.py'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            logger.info("✓ Models trained successfully")
            
            # Check if model files were created
            model_files = [
                'models/patient_to_specialization_model.pkl',
                'models/doctor_disease_model.pkl',
                'models/patient_nlp_pipeline.pkl',
                'models/doctor_nlp_pipeline.pkl'
            ]
            
            created_models = 0
            for model_file in model_files:
                if os.path.exists(model_file):
                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    logger.info(f"  ✓ {model_file} ({size_mb:.1f} MB)")
                    created_models += 1
            
            logger.info(f"✓ {created_models}/{len(model_files)} model files created")
            return True
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Model training timed out (10 minutes)")
        return False
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def start_ml_api():
    """Start the ML API server"""
    logger.info("Starting ML API server...")
    
    try:
        # Start the API server in background
        process = subprocess.Popen([
            sys.executable, 'api/ml_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a few seconds for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✓ ML API server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"ML API server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"Error starting ML API server: {e}")
        return None

def test_ml_api():
    """Test the ML API endpoints"""
    logger.info("Testing ML API endpoints...")
    
    base_url = "http://localhost:5001"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✓ Health endpoint working")
        else:
            logger.error(f"Health endpoint failed: {response.status_code}")
            return False
        
        # Test specialization prediction
        test_data = {
            "symptoms": "chest pain and shortness of breath",
            "top_k": 3
        }
        
        response = requests.post(
            f"{base_url}/predict/specialization",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'specializations' in result:
                logger.info(f"✓ Specialization prediction working: {len(result['specializations'])} results")
            else:
                logger.warning("Specialization prediction returned unexpected format")
        else:
            logger.error(f"Specialization prediction failed: {response.status_code}")
            return False
        
        # Test diagnosis prediction
        test_data = {
            "symptoms": "patient presents with acute chest pain and diaphoresis",
            "top_diseases": 3,
            "top_medicines": 3
        }
        
        response = requests.post(
            f"{base_url}/predict/diagnosis",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'diseases' in result and 'medicines' in result:
                logger.info(f"✓ Diagnosis prediction working: {len(result['diseases'])} diseases, {len(result['medicines'])} medicines")
            else:
                logger.warning("Diagnosis prediction returned unexpected format")
        else:
            logger.error(f"Diagnosis prediction failed: {response.status_code}")
            return False
        
        logger.info("✓ All API endpoints working correctly")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API test failed: {e}")
        return False

def generate_setup_report():
    """Generate setup completion report"""
    logger.info("Generating setup report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'setup_status': 'completed',
        'components': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'dependencies_installed': True,
            'nltk_data_downloaded': True,
            'datasets_available': check_datasets(),
            'models_trained': os.path.exists('models/patient_to_specialization_model.pkl'),
            'api_server_running': True
        },
        'endpoints': {
            'health': 'http://localhost:5001/health',
            'specialization_prediction': 'http://localhost:5001/predict/specialization',
            'diagnosis_prediction': 'http://localhost:5001/predict/diagnosis',
            'models_info': 'http://localhost:5001/models/info'
        },
        'integration': {
            'spring_boot_endpoints': [
                '/api/ml/predict/patient-specialization',
                '/api/ml/predict/doctor-diagnosis',
                '/api/ml/api-health'
            ],
            'frontend_components': [
                'PatientSymptomAnalyzer.jsx',
                'DoctorDiagnosisAssistant.jsx'
            ]
        }
    }
    
    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/setup_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("✓ Setup report saved to: results/setup_report.json")
    return report

def print_success_message():
    """Print setup success message"""
    success_message = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎉 SETUP COMPLETED! 🎉                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  ✅ MedReserve AI ML System is ready!                       ║
    ║                                                              ║
    ║  🚀 ML API Server: http://localhost:5001                    ║
    ║  📊 Health Check: http://localhost:5001/health              ║
    ║  🧠 Models: Patient → Specialization, Doctor → Diagnosis    ║
    ║                                                              ║
    ║  📋 Next Steps:                                              ║
    ║  1. Keep ML API server running                               ║
    ║  2. Start Spring Boot backend                                ║
    ║  3. Test frontend ML components                              ║
    ║  4. Monitor API health and performance                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(success_message)

def main():
    """Main setup function"""
    print_banner()
    
    logger.info("Starting MedReserve AI ML System setup...")
    
    # Step 1: Check Python version
    if not check_python_version():
        logger.error("Setup failed: Incompatible Python version")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        logger.error("Setup failed: Could not install dependencies")
        return False
    
    # Step 3: Download NLTK data
    if not download_nltk_data():
        logger.error("Setup failed: Could not download NLTK data")
        return False
    
    # Step 4: Check datasets
    datasets_available = check_datasets()
    if not datasets_available:
        logger.warning("Limited datasets available - fallback predictions will be used")
    
    # Step 5: Train models (if datasets available)
    if datasets_available:
        if not train_models():
            logger.warning("Model training failed - fallback predictions will be used")
    else:
        logger.info("Skipping model training - no datasets available")
    
    # Step 6: Start ML API server
    api_process = start_ml_api()
    if not api_process:
        logger.error("Setup failed: Could not start ML API server")
        return False
    
    # Step 7: Test API endpoints
    if not test_ml_api():
        logger.error("Setup failed: API tests failed")
        if api_process:
            api_process.terminate()
        return False
    
    # Step 8: Generate setup report
    report = generate_setup_report()
    
    # Step 9: Print success message
    print_success_message()
    
    logger.info("🎉 MedReserve AI ML System setup completed successfully!")
    logger.info("ML API server is running at http://localhost:5001")
    logger.info("Press Ctrl+C to stop the server")
    
    # Keep the API server running
    try:
        api_process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down ML API server...")
        api_process.terminate()
        api_process.wait()
        logger.info("ML API server stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
