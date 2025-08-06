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

        # Set NLTK data path to app directory
        nltk_data_dir = "/app/nltk_data"
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)  # Insert at beginning for priority

        # Download essential data
        essential_data = ['punkt', 'stopwords', 'wordnet']
        optional_data = ['punkt_tab']  # May not exist in all NLTK versions

        success_count = 0
        for data_name in essential_data:
            try:
                nltk.download(data_name, download_dir=nltk_data_dir, quiet=True)
                success_count += 1
                logger.info(f"✅ Downloaded {data_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {data_name}: {e}")

        # Try optional data
        for data_name in optional_data:
            try:
                nltk.download(data_name, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"✅ Downloaded {data_name}")
            except Exception as e:
                logger.info(f"ℹ️ Optional data {data_name} not available: {e}")

        if success_count >= 2:  # At least punkt and stopwords
            logger.info("✅ NLTK data downloaded successfully!")
            return True
        else:
            logger.warning("⚠️ Some NLTK data failed to download")
            return False

    except Exception as e:
        logger.error(f"❌ Error downloading NLTK data: {e}")
        logger.info("Continuing without NLTK data - some features may be limited")
        return False

def start_api_server():
    """Start the Flask API server with fallback support"""
    logger.info("🚀 Starting MedReserve ML API server...")

    # Get port from environment (Render uses PORT, default to 5001)
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Try to start the full ML API first
    try:
        # Set up environment
        os.environ['PYTHONPATH'] = '/app'
        sys.path.insert(0, '/app')

        logger.info("Attempting to start full ML API...")
        from api.ml_api import app, initialize_models

        # Initialize models
        initialize_models()

        logger.info(f"✅ Starting full ML API server on 0.0.0.0:{port}")
        logger.info(f"Debug mode: {debug}")

        app.run(host='0.0.0.0', port=port, debug=debug)

    except Exception as e:
        logger.error(f"❌ Full ML API failed: {e}")
        logger.info("🔄 Falling back to simple ML API...")

        try:
            # Import and run the simple ML API as fallback
            from api.simple_ml_api import app as simple_app

            logger.info(f"✅ Starting simple ML API server on 0.0.0.0:{port}")
            logger.info("Mode: Simple fallback (rule-based predictions)")

            simple_app.run(host='0.0.0.0', port=port, debug=debug)

        except Exception as fallback_error:
            logger.error(f"❌ Even simple API failed: {fallback_error}")
            import traceback
            logger.error(traceback.format_exc())
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
