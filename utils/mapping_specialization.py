"""
Disease to Doctor Specialization Mapping for MedReserve AI
Maps diseases to appropriate medical specializations
"""

import pandas as pd
from typing import Dict, List, Tuple
import re

class DiseaseSpecializationMapper:
    """
    Maps diseases to appropriate doctor specializations
    """
    
    def __init__(self):
        # Comprehensive disease to specialization mapping
        self.disease_specialization_map = {
            # Cardiology
            'heart attack': 'Cardiology',
            'myocardial infarction': 'Cardiology',
            'angina': 'Cardiology',
            'arrhythmia': 'Cardiology',
            'atrial fibrillation': 'Cardiology',
            'heart failure': 'Cardiology',
            'coronary artery disease': 'Cardiology',
            'hypertension': 'Cardiology',
            'high blood pressure': 'Cardiology',
            'cardiac arrest': 'Cardiology',
            'pericarditis': 'Cardiology',
            'endocarditis': 'Cardiology',
            'valve disease': 'Cardiology',
            
            # Neurology
            'stroke': 'Neurology',
            'epilepsy': 'Neurology',
            'seizure': 'Neurology',
            'migraine': 'Neurology',
            'headache': 'Neurology',
            'alzheimer': 'Neurology',
            'dementia': 'Neurology',
            'parkinson': 'Neurology',
            'multiple sclerosis': 'Neurology',
            'neuropathy': 'Neurology',
            'brain tumor': 'Neurology',
            'meningitis': 'Neurology',
            'encephalitis': 'Neurology',
            'concussion': 'Neurology',
            'vertigo': 'Neurology',
            
            # Pulmonology
            'asthma': 'Pulmonology',
            'copd': 'Pulmonology',
            'pneumonia': 'Pulmonology',
            'bronchitis': 'Pulmonology',
            'tuberculosis': 'Pulmonology',
            'lung cancer': 'Pulmonology',
            'pulmonary embolism': 'Pulmonology',
            'sleep apnea': 'Pulmonology',
            'emphysema': 'Pulmonology',
            'pleurisy': 'Pulmonology',
            'respiratory failure': 'Pulmonology',
            
            # Gastroenterology
            'gastritis': 'Gastroenterology',
            'ulcer': 'Gastroenterology',
            'peptic ulcer': 'Gastroenterology',
            'gerd': 'Gastroenterology',
            'acid reflux': 'Gastroenterology',
            'ibs': 'Gastroenterology',
            'irritable bowel syndrome': 'Gastroenterology',
            'crohn disease': 'Gastroenterology',
            'ulcerative colitis': 'Gastroenterology',
            'hepatitis': 'Gastroenterology',
            'cirrhosis': 'Gastroenterology',
            'pancreatitis': 'Gastroenterology',
            'gallstones': 'Gastroenterology',
            'appendicitis': 'Gastroenterology',
            'diverticulitis': 'Gastroenterology',
            'colon cancer': 'Gastroenterology',
            
            # Endocrinology
            'diabetes': 'Endocrinology',
            'type 1 diabetes': 'Endocrinology',
            'type 2 diabetes': 'Endocrinology',
            'thyroid': 'Endocrinology',
            'hyperthyroidism': 'Endocrinology',
            'hypothyroidism': 'Endocrinology',
            'goiter': 'Endocrinology',
            'adrenal insufficiency': 'Endocrinology',
            'cushing syndrome': 'Endocrinology',
            'pituitary disorder': 'Endocrinology',
            'metabolic syndrome': 'Endocrinology',
            'obesity': 'Endocrinology',
            
            # Dermatology
            'eczema': 'Dermatology',
            'psoriasis': 'Dermatology',
            'acne': 'Dermatology',
            'dermatitis': 'Dermatology',
            'skin cancer': 'Dermatology',
            'melanoma': 'Dermatology',
            'rash': 'Dermatology',
            'hives': 'Dermatology',
            'fungal infection': 'Dermatology',
            'warts': 'Dermatology',
            'vitiligo': 'Dermatology',
            'alopecia': 'Dermatology',
            
            # Orthopedics
            'fracture': 'Orthopedics',
            'arthritis': 'Orthopedics',
            'osteoarthritis': 'Orthopedics',
            'rheumatoid arthritis': 'Rheumatology',
            'back pain': 'Orthopedics',
            'joint pain': 'Orthopedics',
            'osteoporosis': 'Orthopedics',
            'tendonitis': 'Orthopedics',
            'bursitis': 'Orthopedics',
            'carpal tunnel': 'Orthopedics',
            'herniated disc': 'Orthopedics',
            'scoliosis': 'Orthopedics',
            
            # Psychiatry
            'depression': 'Psychiatry',
            'anxiety': 'Psychiatry',
            'bipolar disorder': 'Psychiatry',
            'schizophrenia': 'Psychiatry',
            'ptsd': 'Psychiatry',
            'ocd': 'Psychiatry',
            'adhd': 'Psychiatry',
            'eating disorder': 'Psychiatry',
            'panic disorder': 'Psychiatry',
            'substance abuse': 'Psychiatry',
            
            # Gynecology
            'endometriosis': 'Gynecology',
            'pcos': 'Gynecology',
            'ovarian cyst': 'Gynecology',
            'uterine fibroids': 'Gynecology',
            'menstrual disorder': 'Gynecology',
            'pelvic inflammatory disease': 'Gynecology',
            'cervical cancer': 'Gynecology',
            'ovarian cancer': 'Gynecology',
            'pregnancy': 'Obstetrics',
            
            # Urology
            'kidney stones': 'Urology',
            'uti': 'Urology',
            'urinary tract infection': 'Urology',
            'prostate': 'Urology',
            'bladder infection': 'Urology',
            'kidney disease': 'Nephrology',
            'erectile dysfunction': 'Urology',
            'incontinence': 'Urology',
            
            # Ophthalmology
            'glaucoma': 'Ophthalmology',
            'cataract': 'Ophthalmology',
            'macular degeneration': 'Ophthalmology',
            'diabetic retinopathy': 'Ophthalmology',
            'eye infection': 'Ophthalmology',
            'vision loss': 'Ophthalmology',
            'dry eyes': 'Ophthalmology',
            
            # ENT (Otolaryngology)
            'sinusitis': 'ENT',
            'tonsillitis': 'ENT',
            'hearing loss': 'ENT',
            'ear infection': 'ENT',
            'throat infection': 'ENT',
            'nasal polyps': 'ENT',
            'deviated septum': 'ENT',
            'laryngitis': 'ENT',
            
            # Infectious Disease
            'malaria': 'Infectious Disease',
            'dengue': 'Infectious Disease',
            'typhoid': 'Infectious Disease',
            'hiv': 'Infectious Disease',
            'covid': 'Infectious Disease',
            'influenza': 'Internal Medicine',
            'common cold': 'Internal Medicine',
            'food poisoning': 'Gastroenterology',
            
            # General/Internal Medicine (default for common conditions)
            'fever': 'Internal Medicine',
            'fatigue': 'Internal Medicine',
            'weight loss': 'Internal Medicine',
            'weight gain': 'Internal Medicine',
            'dizziness': 'Internal Medicine',
            'nausea': 'Internal Medicine',
            'vomiting': 'Internal Medicine',
            'diarrhea': 'Gastroenterology',
            'constipation': 'Gastroenterology',
            'abdominal pain': 'Gastroenterology',
            'chest pain': 'Cardiology',
            'shortness of breath': 'Pulmonology',
            'cough': 'Pulmonology',
        }
        
        # Specialization priority (for when multiple specializations are possible)
        self.specialization_priority = {
            'Cardiology': 10,
            'Neurology': 9,
            'Pulmonology': 8,
            'Gastroenterology': 7,
            'Endocrinology': 6,
            'Dermatology': 5,
            'Orthopedics': 4,
            'Psychiatry': 3,
            'Internal Medicine': 2,
            'General Practice': 1
        }
    
    def get_specialization(self, disease: str) -> str:
        """
        Get the most appropriate specialization for a disease
        """
        disease_lower = disease.lower().strip()
        
        # Direct mapping
        if disease_lower in self.disease_specialization_map:
            return self.disease_specialization_map[disease_lower]
        
        # Partial matching for compound diseases
        for disease_key, specialization in self.disease_specialization_map.items():
            if disease_key in disease_lower or disease_lower in disease_key:
                return specialization
        
        # Default to Internal Medicine for unknown diseases
        return 'Internal Medicine'
    
    def get_top_specializations(self, diseases: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K specializations based on predicted diseases
        """
        specialization_scores = {}
        
        for disease in diseases:
            spec = self.get_specialization(disease)
            priority = self.specialization_priority.get(spec, 1)
            
            if spec in specialization_scores:
                specialization_scores[spec] += priority
            else:
                specialization_scores[spec] = priority
        
        # Sort by score and return top K
        sorted_specs = sorted(
            specialization_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Normalize scores to probabilities
        total_score = sum(score for _, score in sorted_specs)
        normalized_specs = [
            (spec, score / total_score) 
            for spec, score in sorted_specs[:top_k]
        ]
        
        return normalized_specs
    
    def load_specialization_data(self, filepath: str) -> pd.DataFrame:
        """
        Load additional specialization data from CSV file
        """
        try:
            df = pd.read_csv(filepath)
            
            # Update mapping from loaded data
            if 'disease' in df.columns and 'specialization' in df.columns:
                for _, row in df.iterrows():
                    disease = str(row['disease']).lower().strip()
                    spec = str(row['specialization']).strip()
                    self.disease_specialization_map[disease] = spec
            
            return df
        except Exception as e:
            print(f"Error loading specialization data: {e}")
            return pd.DataFrame()
    
    def get_all_specializations(self) -> List[str]:
        """
        Get list of all available specializations
        """
        return list(set(self.disease_specialization_map.values()))
    
    def search_diseases_by_specialization(self, specialization: str) -> List[str]:
        """
        Get all diseases handled by a specific specialization
        """
        diseases = []
        for disease, spec in self.disease_specialization_map.items():
            if spec.lower() == specialization.lower():
                diseases.append(disease)
        return diseases

# Example usage and testing
if __name__ == "__main__":
    mapper = DiseaseSpecializationMapper()
    
    # Test individual disease mapping
    test_diseases = [
        'diabetes', 'heart attack', 'asthma', 'depression', 
        'skin rash', 'back pain', 'migraine'
    ]
    
    print("Disease to Specialization Mapping:")
    for disease in test_diseases:
        spec = mapper.get_specialization(disease)
        print(f"- {disease} → {spec}")
    
    # Test top specializations for multiple diseases
    predicted_diseases = ['diabetes', 'hypertension', 'chest pain']
    top_specs = mapper.get_top_specializations(predicted_diseases)
    
    print(f"\nTop specializations for {predicted_diseases}:")
    for spec, score in top_specs:
        print(f"- {spec}: {score:.2f}")
    
    # Show all available specializations
    all_specs = mapper.get_all_specializations()
    print(f"\nAll available specializations ({len(all_specs)}):")
    for spec in sorted(all_specs):
        print(f"- {spec}")
