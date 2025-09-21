# 🧠 MedReserve — ML Service (Flask)

See overall architecture diagram: [../../docs/architecture.mmd](../../docs/architecture.mmd)

REST API that provides:
- Patient → specialization recommendations
- Doctor → diagnosis and medicine suggestions

Runs on http://localhost:5001 by default.

## 🚀 Quickstart

OS-specific quick reference: see the root README section “OS-specific instructions”.
```
cd backend/ml
pip install -r requirements.txt
python api/ml_api.py
```

Health: GET http://localhost:5001/health

## 📡 Endpoints (JSON)
- POST /predict/specialization
  - { "symptoms": "chest pain and shortness of breath", "top_k": 3 }
- POST /predict/diagnosis
  - { "symptoms": "...", "top_diseases": 5, "top_medicines": 5 }
- POST /predict/batch/specialization
- POST /predict/batch/diagnosis
- GET /models/info

## 🔧 Configuration (.env example — no secrets)
```
PORT=5001
DEBUG=false
```

## 🧪 Testing
```
pytest -q
# Smoke
curl http://localhost:5001/health
```

## 🐳 Docker (service-only)
```
docker build -t medreserve-ml .
docker run -p 5001:5001 medreserve-ml
```

Tip: Prefer running all services with a root docker-compose (ask me to create it).

## 🌐 Production
- Example ML URL: https://medreserve-ml.onrender.com (update if different)

## 🔒 Notes
- Input is validated; avoid sending PII
- Use HTTPS in production
