"""
Setup script for Disease Prediction Pipeline.
Handles installation, model training, and testing.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description or command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n🔧 Installing Dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install basic requirements")
        return False
    
    # Install spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️ Warning: spaCy model download failed. Text preprocessing may be limited.")
    
    print("✓ Dependencies installed successfully")
    return True

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = ["data", "models", "preprocessing"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")
    
    return True

def check_dataset():
    """Check if dataset exists and validate format."""
    print("\n📊 Checking dataset...")
    
    dataset_path = "data/symptoms_dataset.csv"
    sample_dataset_path = "data/sample_symptoms_dataset.csv"
    
    if os.path.exists(dataset_path):
        print(f"✓ Found dataset at {dataset_path}")
        
        # Basic validation
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            
            if 'symptoms' in df.columns and 'disease' in df.columns:
                print(f"✓ Dataset format is valid")
                print(f"✓ Dataset contains {len(df)} samples")
                print(f"✓ Dataset contains {df['disease'].nunique()} unique diseases")
                return True
            else:
                print("❌ Dataset must have 'symptoms' and 'disease' columns")
                return False
                
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    
    elif os.path.exists(sample_dataset_path):
        print(f"⚠️ Main dataset not found, but sample dataset exists at {sample_dataset_path}")
        print("You can use the sample dataset for testing or replace it with your own data")
        
        # Copy sample to main dataset for testing
        shutil.copy(sample_dataset_path, dataset_path)
        print(f"✓ Copied sample dataset to {dataset_path} for testing")
        return True
    
    else:
        print(f"❌ No dataset found at {dataset_path}")
        print("Please place your dataset file with columns: symptoms,disease")
        return False

def train_models():
    """Train both ML and DL models."""
    print("\n🤖 Training Models...")
    
    # Train ML model
    print("\n1. Training Machine Learning Model...")
    if not run_command("python train_ml_model.py", "Training ML model"):
        print("❌ ML model training failed")
        return False
    
    # Train DL model
    print("\n2. Training Deep Learning Model...")
    if not run_command("python train_dl_model.py", "Training DL model"):
        print("❌ DL model training failed")
        return False
    
    print("✓ Both models trained successfully")
    return True

def test_predictions():
    """Test the prediction pipeline."""
    print("\n🧪 Testing Predictions...")
    
    test_symptoms = [
        "I have high fever and body pain",
        "Experiencing cough and cold symptoms",
        "Stomach ache with nausea"
    ]
    
    try:
        # Test ML predictions
        print("\nTesting ML predictions...")
        from predict_ml import predict_with_ml
        
        for symptom in test_symptoms:
            result = predict_with_ml(symptom)
            if "error" not in result:
                print(f"✓ ML: '{symptom}' -> {result['predicted_disease']} ({result['confidence']:.3f})")
            else:
                print(f"❌ ML Error: {result['error']}")
        
        # Test DL predictions
        print("\nTesting DL predictions...")
        from predict_dl import predict_with_dl
        
        for symptom in test_symptoms:
            result = predict_with_dl(symptom)
            if "error" not in result:
                print(f"✓ DL: '{symptom}' -> {result['predicted_disease']} ({result['confidence']:.3f})")
            else:
                print(f"❌ DL Error: {result['error']}")
        
        # Test ensemble predictions
        print("\nTesting Ensemble predictions...")
        from ensemble_predictor import predict_disease
        
        for symptom in test_symptoms:
            result = predict_disease(symptom)
            if "error" not in result:
                print(f"✓ Ensemble: '{symptom}' -> {result['predicted_disease']} ({result['confidence']:.3f})")
            else:
                print(f"❌ Ensemble Error: {result['error']}")
        
        print("✓ All prediction tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🏥 Disease Prediction Pipeline Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup directories
    if not setup_directories():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check dataset
    if not check_dataset():
        print("\n⚠️ Dataset issues detected. Please fix before training models.")
        return False
    
    # Train models
    train_choice = input("\n🤖 Do you want to train the models now? (y/n): ").lower().strip()
    if train_choice in ['y', 'yes']:
        if not train_models():
            print("\n❌ Model training failed. Please check the errors above.")
            return False
        
        # Test predictions
        if not test_predictions():
            print("\n❌ Prediction testing failed.")
            return False
    else:
        print("\n⏭️ Skipping model training. You can train later using:")
        print("   python train_ml_model.py")
        print("   python train_dl_model.py")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Place your dataset at data/symptoms_dataset.csv")
    print("2. Train models: python train_ml_model.py && python train_dl_model.py")
    print("3. Test predictions: python predict_ml.py or python predict_dl.py")
    print("4. Use ensemble: python ensemble_predictor.py")
    print("\n📖 For more information, see README_disease_prediction.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
