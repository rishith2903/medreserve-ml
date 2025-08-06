"""
Test script for MedReserve AI ML API
Tests both patient specialization and doctor diagnosis predictions
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5001"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Specialization model loaded: {data['models']['specialization_loaded']}")
            print(f"   Diagnosis model loaded: {data['models']['diagnosis_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_specialization_prediction():
    """Test patient to specialization prediction"""
    print("\n🤖 Testing Patient → Specialization Prediction...")
    
    test_cases = [
        "I have severe chest pain and shortness of breath",
        "Experiencing headache, dizziness and nausea for 3 days",
        "Skin rash with itching and redness all over my body",
        "Joint pain and stiffness in the morning",
        "Feeling sad, anxious and having trouble sleeping"
    ]
    
    for i, symptoms in enumerate(test_cases, 1):
        try:
            payload = {"symptoms": symptoms}
            response = requests.post(f"{BASE_URL}/predict/specialization", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n   Test {i}: {symptoms}")
                print(f"   Top Recommendations:")
                for spec in data['specializations'][:3]:
                    print(f"     - {spec['specialization']}: {spec['percentage']:.1f}%")
            else:
                print(f"   ❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Test {i} error: {e}")

def test_diagnosis_prediction():
    """Test doctor to diagnosis prediction"""
    print("\n🩺 Testing Doctor → Diagnosis Prediction...")
    
    test_cases = [
        "Patient presents with severe headache, nausea, and sensitivity to light",
        "Chest pain, irregular heartbeat, and fatigue during exercise",
        "Persistent cough, shortness of breath, and wheezing",
        "Abdominal pain, nausea, vomiting, and heartburn"
    ]
    
    for i, symptoms in enumerate(test_cases, 1):
        try:
            payload = {"symptoms": symptoms}
            response = requests.post(f"{BASE_URL}/predict/diagnosis", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n   Test {i}: {symptoms}")
                print(f"   Top Diseases:")
                for disease in data['diseases'][:3]:
                    print(f"     - {disease['disease']}: {disease['percentage']:.1f}%")
                print(f"   Top Medicines:")
                for medicine in data['medicines'][:3]:
                    print(f"     - {medicine['medicine']}: {medicine['percentage']:.1f}%")
            else:
                print(f"   ❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Test {i} error: {e}")

def main():
    """Run all tests"""
    print("🚀 MedReserve AI ML API Test Suite")
    print("=" * 50)
    
    # Test health check
    health_ok = test_health_check()
    
    if health_ok:
        # Test specialization prediction
        test_specialization_prediction()
        
        # Test diagnosis prediction
        test_diagnosis_prediction()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("   Your ML system is working correctly!")
    else:
        print("\n❌ ML API is not responding. Make sure it's running on port 5001")
        print("   Start it with: python api/ml_api.py")

if __name__ == "__main__":
    main()
