#!/usr/bin/env python3
"""
MedReserve ML Service Startup Script
Ensures models are trained and starts the Flask API server
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_models_exist():
    """Check if trained models exist"""
    models_dir = Path("models")
    required_models = [
        "patient_to_specialization_model.pkl",
        "doctor_disease_model.pkl",
        "doctor_medicine_model.pkl"
    ]
    
    if not models_dir.exists():
        return False
    
    for model_file in required_models:
        if not (models_dir / model_file).exists():
            return False
    
    return True

def train_models():
    """Train ML models if they don't exist"""
    logger.info("🤖 Training ML models...")
    try:
        result = subprocess.run([
            sys.executable, "train_all_models.py"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Models trained successfully!")
            return True
        else:
            logger.error(f"❌ Model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Model training timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading NLTK data: {e}")
        return False

def start_api_server():
    """Start the Flask API server"""
    logger.info("🚀 Starting MedReserve ML API server...")
    try:
        # Import and run the ML API
        from api.ml_api import app, initialize_models
        
        # Initialize models
        initialize_models()
        
        # Get port from environment
        port = int(os.environ.get('PORT', 5001))
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"❌ Error starting API server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("🏥 Starting MedReserve ML Service")
    logger.info("=" * 50)
    
    # Download NLTK data
    if not download_nltk_data():
        logger.warning("⚠️ NLTK data download failed, continuing anyway...")
    
    # Check if models exist, train if necessary
    if not check_models_exist():
        logger.info("📊 Models not found, training new models...")
        if not train_models():
            logger.error("❌ Failed to train models, starting with fallback mode...")
    else:
        logger.info("✅ Models found, skipping training")
    
    # Start the API server
    start_api_server()

if __name__ == "__main__":
    main()
