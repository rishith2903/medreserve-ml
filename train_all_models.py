"""
Train All MedReserve AI Models
Generates sample medical data and trains both patient and doctor models
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.train_patient_model import PatientSpecializationModel
from train.train_doctor_model import DoctorDiagnosisModel

def create_sample_medical_data():
    """
    Create comprehensive sample medical datasets for training
    """
    print("🏥 Creating sample medical datasets...")
    
    # Sample diseases and their symptoms
    medical_data = [
        # Cardiology
        ("Chest pain, shortness of breath, fatigue", "Heart Disease", "Cardiology", "Aspirin, Beta-blockers"),
        ("Irregular heartbeat, dizziness, chest discomfort", "Arrhythmia", "Cardiology", "Antiarrhythmic drugs"),
        ("High blood pressure, headache, blurred vision", "Hypertension", "Cardiology", "ACE inhibitors, Diuretics"),
        ("Chest pain during exercise, shortness of breath", "Angina", "Cardiology", "Nitroglycerin, Beta-blockers"),
        
        # Neurology
        ("Severe headache, nausea, sensitivity to light", "Migraine", "Neurology", "Sumatriptan, Pain relievers"),
        ("Memory loss, confusion, difficulty speaking", "Dementia", "Neurology", "Cholinesterase inhibitors"),
        ("Tremor, stiffness, slow movement", "Parkinson's Disease", "Neurology", "Levodopa, Dopamine agonists"),
        ("Seizures, loss of consciousness, confusion", "Epilepsy", "Neurology", "Anticonvulsants"),
        
        # Dermatology
        ("Red, itchy, scaly skin patches", "Eczema", "Dermatology", "Topical corticosteroids, Moisturizers"),
        ("Red, raised, scaly patches on skin", "Psoriasis", "Dermatology", "Topical treatments, Immunosuppressants"),
        ("Painful, fluid-filled blisters", "Herpes", "Dermatology", "Antiviral medications"),
        ("Itchy, red, swollen skin", "Allergic Reaction", "Dermatology", "Antihistamines, Corticosteroids"),
        
        # Orthopedics
        ("Joint pain, stiffness, swelling", "Arthritis", "Orthopedics", "NSAIDs, Physical therapy"),
        ("Back pain, muscle spasms, limited mobility", "Back Pain", "Orthopedics", "Pain relievers, Muscle relaxants"),
        ("Bone pain, fracture, swelling", "Bone Fracture", "Orthopedics", "Pain medication, Immobilization"),
        ("Knee pain, swelling, difficulty walking", "Knee Injury", "Orthopedics", "NSAIDs, Physical therapy"),
        
        # Gastroenterology
        ("Abdominal pain, nausea, vomiting", "Gastritis", "Gastroenterology", "Proton pump inhibitors, Antacids"),
        ("Heartburn, acid reflux, chest pain", "GERD", "Gastroenterology", "Proton pump inhibitors"),
        ("Diarrhea, abdominal cramps, fever", "Food Poisoning", "Gastroenterology", "Fluids, Electrolytes"),
        ("Constipation, abdominal pain, bloating", "IBS", "Gastroenterology", "Fiber supplements, Antispasmodics"),
        
        # Pulmonology
        ("Cough, shortness of breath, wheezing", "Asthma", "Pulmonology", "Bronchodilators, Corticosteroids"),
        ("Persistent cough, chest pain, fever", "Pneumonia", "Pulmonology", "Antibiotics, Rest"),
        ("Chronic cough, mucus production, fatigue", "COPD", "Pulmonology", "Bronchodilators, Oxygen therapy"),
        ("Sudden chest pain, difficulty breathing", "Pneumothorax", "Pulmonology", "Chest tube, Surgery"),
        
        # Endocrinology
        ("Increased thirst, frequent urination, fatigue", "Diabetes", "Endocrinology", "Insulin, Metformin"),
        ("Weight gain, fatigue, cold intolerance", "Hypothyroidism", "Endocrinology", "Levothyroxine"),
        ("Weight loss, rapid heartbeat, anxiety", "Hyperthyroidism", "Endocrinology", "Antithyroid medications"),
        ("Excessive hunger, weight loss, fatigue", "Type 1 Diabetes", "Endocrinology", "Insulin therapy"),
        
        # Psychiatry
        ("Persistent sadness, loss of interest, fatigue", "Depression", "Psychiatry", "Antidepressants, Therapy"),
        ("Excessive worry, restlessness, panic attacks", "Anxiety", "Psychiatry", "Anxiolytics, Therapy"),
        ("Mood swings, manic episodes, depression", "Bipolar Disorder", "Psychiatry", "Mood stabilizers"),
        ("Hallucinations, delusions, disorganized thinking", "Schizophrenia", "Psychiatry", "Antipsychotics"),
        
        # Ophthalmology
        ("Blurred vision, eye pain, halos around lights", "Glaucoma", "Ophthalmology", "Eye drops, Surgery"),
        ("Cloudy vision, difficulty seeing at night", "Cataracts", "Ophthalmology", "Surgery"),
        ("Red, itchy, watery eyes", "Conjunctivitis", "Ophthalmology", "Antibiotic drops"),
        ("Sudden vision loss, eye pain", "Retinal Detachment", "Ophthalmology", "Emergency surgery"),
        
        # ENT (Otolaryngology)
        ("Sore throat, difficulty swallowing, fever", "Strep Throat", "ENT", "Antibiotics, Pain relievers"),
        ("Ear pain, hearing loss, discharge", "Ear Infection", "ENT", "Antibiotics, Pain relievers"),
        ("Nasal congestion, facial pain, headache", "Sinusitis", "ENT", "Decongestants, Antibiotics"),
        ("Hoarse voice, throat pain, cough", "Laryngitis", "ENT", "Voice rest, Humidifier"),
        
        # Urology
        ("Painful urination, frequent urination, urgency", "UTI", "Urology", "Antibiotics, Fluids"),
        ("Blood in urine, flank pain, nausea", "Kidney Stones", "Urology", "Pain medication, Fluids"),
        ("Difficulty urinating, weak stream", "Prostate Problems", "Urology", "Alpha blockers"),
        ("Pelvic pain, urinary incontinence", "Bladder Issues", "Urology", "Anticholinergics"),
        
        # Gynecology
        ("Irregular periods, pelvic pain, heavy bleeding", "Menstrual Disorders", "Gynecology", "Hormonal therapy"),
        ("Pelvic pain, painful periods, infertility", "Endometriosis", "Gynecology", "Hormonal therapy, Surgery"),
        ("Vaginal discharge, itching, burning", "Yeast Infection", "Gynecology", "Antifungal medication"),
        ("Missed period, nausea, breast tenderness", "Pregnancy", "Gynecology", "Prenatal vitamins"),
        
        # Pediatrics
        ("Fever, runny nose, cough in child", "Common Cold", "Pediatrics", "Rest, Fluids, Fever reducers"),
        ("Rash, fever, sore throat in child", "Viral Infection", "Pediatrics", "Supportive care"),
        ("Ear pain, fever, irritability in child", "Pediatric Ear Infection", "Pediatrics", "Antibiotics"),
        ("Stomach pain, vomiting, diarrhea in child", "Gastroenteritis", "Pediatrics", "Fluids, Rest"),
        
        # General Medicine
        ("Fever, body aches, fatigue", "Flu", "General Medicine", "Rest, Fluids, Antivirals"),
        ("Runny nose, sneezing, mild fever", "Common Cold", "General Medicine", "Rest, Fluids"),
        ("Fatigue, weakness, pale skin", "Anemia", "General Medicine", "Iron supplements"),
        ("High fever, chills, body aches", "Infection", "General Medicine", "Antibiotics, Rest"),

        # Additional samples for better training
        ("Chest tightness, palpitations", "Heart Palpitations", "Cardiology", "Beta-blockers"),
        ("Leg swelling, shortness of breath", "Heart Failure", "Cardiology", "Diuretics, ACE inhibitors"),
        ("Sudden severe headache, neck stiffness", "Severe Headache", "Neurology", "Pain relievers, Imaging"),
        ("Numbness in hands, tingling", "Neuropathy", "Neurology", "Gabapentin, Vitamin B12"),
        ("Dry skin, itching, redness", "Dry Skin", "Dermatology", "Moisturizers, Topical steroids"),
        ("Hair loss, scalp irritation", "Alopecia", "Dermatology", "Minoxidil, Topical treatments"),
        ("Joint swelling, morning stiffness", "Rheumatoid Arthritis", "Orthopedics", "DMARDs, NSAIDs"),
        ("Muscle weakness, bone pain", "Osteoporosis", "Orthopedics", "Calcium, Vitamin D"),
        ("Acid reflux, stomach pain", "Peptic Ulcer", "Gastroenterology", "Proton pump inhibitors"),
        ("Bloating, gas, stomach cramps", "Digestive Issues", "Gastroenterology", "Probiotics, Digestive enzymes"),
        ("Persistent cough, chest tightness", "Bronchitis", "Pulmonology", "Bronchodilators, Cough suppressants"),
        ("Difficulty breathing, chest pain", "Respiratory Issues", "Pulmonology", "Inhalers, Oxygen therapy"),
        ("Excessive sweating, heat intolerance", "Thyroid Issues", "Endocrinology", "Thyroid medications"),
        ("Frequent infections, slow healing", "Immune System Issues", "Endocrinology", "Immune boosters"),
        ("Mood changes, sleep problems", "Mental Health", "Psychiatry", "Antidepressants, Sleep aids"),
        ("Panic attacks, racing heart", "Panic Disorder", "Psychiatry", "Anti-anxiety medications"),
        ("Eye redness, vision changes", "Eye Problems", "Ophthalmology", "Eye drops, Vision correction"),
        ("Double vision, eye strain", "Vision Problems", "Ophthalmology", "Corrective lenses"),
        ("Hearing loss, ear ringing", "Hearing Problems", "ENT", "Hearing aids, Tinnitus treatment"),
        ("Throat clearing, voice changes", "Voice Problems", "ENT", "Voice therapy, Anti-inflammatories"),
        ("Urinary frequency, burning", "Bladder Infection", "Urology", "Antibiotics, Increased fluids"),
        ("Kidney pain, blood in urine", "Kidney Problems", "Urology", "Pain medication, Antibiotics"),
        ("Irregular cycles, cramping", "Menstrual Problems", "Gynecology", "Hormonal therapy, Pain relievers"),
        ("Pelvic pressure, discomfort", "Pelvic Issues", "Gynecology", "Physical therapy, Hormones"),
        ("Fever in child, loss of appetite", "Childhood Illness", "Pediatrics", "Fever reducers, Fluids"),
        ("Growth concerns, developmental delays", "Developmental Issues", "Pediatrics", "Nutritional support"),
        ("Muscle aches, joint pain", "Fibromyalgia", "General Medicine", "Pain relievers, Exercise"),
        ("Chronic fatigue, weakness", "Chronic Fatigue", "General Medicine", "Energy supplements, Rest")
    ]
    
    # Create DataFrame
    df = pd.DataFrame(medical_data, columns=['symptoms', 'disease', 'specialization', 'medicine'])
    
    # Save datasets
    dataset_dir = "backend/ml/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save main dataset
    df.to_csv(f"{dataset_dir}/medical_training_data.csv", index=False)
    
    # Create patient specialization dataset (symptoms -> specialization)
    patient_df = df[['symptoms', 'specialization']].copy()
    patient_df.to_csv(f"{dataset_dir}/patient_specialization_data.csv", index=False)
    
    # Create doctor diagnosis dataset (symptoms -> disease + medicine)
    doctor_df = df[['symptoms', 'disease', 'medicine']].copy()
    doctor_df.to_csv(f"{dataset_dir}/doctor_diagnosis_data.csv", index=False)
    
    print(f"✅ Created datasets with {len(df)} medical cases")
    print(f"   - Patient specialization: {len(patient_df)} cases")
    print(f"   - Doctor diagnosis: {len(doctor_df)} cases")
    
    return patient_df, doctor_df

def train_patient_model(patient_data):
    """
    Train the patient to specialization model
    """
    print("\n🤖 Training Patient → Specialization Model...")
    
    try:
        model = PatientSpecializationModel()
        
        # Prepare data for training
        X = patient_data['symptoms'].values
        y = patient_data['specialization'].values
        
        # Train model
        accuracy = model.train_model_simple(X, y)
        
        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(model_dir)
        
        print(f"✅ Patient model trained successfully! Accuracy: {accuracy:.2%}")
        return True
        
    except Exception as e:
        print(f"❌ Error training patient model: {e}")
        return False

def train_doctor_model(doctor_data):
    """
    Train the doctor diagnosis model
    """
    print("\n🩺 Training Doctor → Diagnosis Model...")
    
    try:
        model = DoctorDiagnosisModel()
        
        # Prepare data for training
        X = doctor_data['symptoms'].values
        y_disease = doctor_data['disease'].values
        y_medicine = doctor_data['medicine'].values
        
        # Train model
        disease_accuracy, medicine_accuracy = model.train_models_simple(X, y_disease, y_medicine)
        
        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model.save_models(model_dir)
        
        print(f"✅ Doctor model trained successfully!")
        print(f"   - Disease prediction accuracy: {disease_accuracy:.2%}")
        print(f"   - Medicine prediction accuracy: {medicine_accuracy:.2%}")
        return True
        
    except Exception as e:
        print(f"❌ Error training doctor model: {e}")
        return False

def main():
    """
    Main training pipeline
    """
    print("🚀 Starting MedReserve AI Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Create sample data
        patient_data, doctor_data = create_sample_medical_data()
        
        # Train patient model
        patient_success = train_patient_model(patient_data)
        
        # Train doctor model
        doctor_success = train_doctor_model(doctor_data)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 Training Summary:")
        print(f"   Patient Model: {'✅ Success' if patient_success else '❌ Failed'}")
        print(f"   Doctor Model: {'✅ Success' if doctor_success else '❌ Failed'}")
        
        if patient_success and doctor_success:
            print("\n🎉 All models trained successfully!")
            print("   You can now start the ML API server with: python api/ml_api.py")
        else:
            print("\n⚠️  Some models failed to train. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
