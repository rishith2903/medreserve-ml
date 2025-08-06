#!/bin/bash

# Quick deployment fix for MedReserve ML Service
# This script helps fix the deployment issues

echo "🔧 MedReserve ML Deployment Fix"
echo "==============================="

# Check current directory
if [ ! -f "api/ml_api.py" ]; then
    echo "❌ Error: Please run this script from the backend/ml directory"
    exit 1
fi

echo "✅ Found ML service files"

# Option to use simple Dockerfile
read -p "Use simple Dockerfile for faster deployment? (y/n): " use_simple

if [ "$use_simple" = "y" ] || [ "$use_simple" = "Y" ]; then
    echo "📝 Switching to simple Dockerfile..."
    cp Dockerfile.simple Dockerfile
    echo "✅ Using simple Dockerfile"
else
    echo "📝 Using advanced Dockerfile with multi-stage build"
fi

# Update render.yaml to ensure correct port
echo "🔧 Updating render.yaml configuration..."
cat > render.yaml << 'EOF'
services:
  - type: web
    name: medreserve-ml
    env: docker
    plan: free
    autoDeploy: true
    region: singapore
    healthCheckPath: /health
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PORT
        value: 5001
      - key: DEBUG
        value: false
      - key: ENVIRONMENT
        value: production
      - key: PYTHONPATH
        value: /app
      - key: PYTHONUNBUFFERED
        value: 1
      - key: NLTK_DATA
        value: /app/nltk_data
      - key: CORS_ALLOWED_ORIGINS
        value: https://med-reserve-ai.vercel.app,https://rishith2903.github.io,http://localhost:3000,https://medreserve-backend.onrender.com
EOF

echo "✅ Updated render.yaml"

# Create a simple health check script
echo "🏥 Creating health check script..."
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
"""Simple health check for ML service"""

import requests
import sys
import time

def check_health(url="http://localhost:5001/health", max_retries=5):
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Health check passed: {response.json()}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Health check attempt {i+1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                time.sleep(5)
    
    return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001/health"
    success = check_health(url)
    sys.exit(0 if success else 1)
EOF

chmod +x health_check.py
echo "✅ Created health check script"

# Test local build (optional)
read -p "Test Docker build locally? (y/n): " test_build

if [ "$test_build" = "y" ] || [ "$test_build" = "Y" ]; then
    echo "🐳 Testing Docker build..."
    docker build -t medreserve-ml-test . 2>&1 | tail -20
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker build successful"
        
        read -p "Test run locally? (y/n): " test_run
        if [ "$test_run" = "y" ] || [ "$test_run" = "Y" ]; then
            echo "🚀 Starting test container..."
            docker run -d -p 5001:5001 --name medreserve-ml-test medreserve-ml-test
            
            echo "⏳ Waiting for service to start..."
            sleep 10
            
            python health_check.py
            
            echo "🛑 Stopping test container..."
            docker stop medreserve-ml-test
            docker rm medreserve-ml-test
        fi
    else
        echo "❌ Docker build failed"
    fi
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Commit and push changes:"
echo "   git add ."
echo "   git commit -m 'Fix ML deployment issues'"
echo "   git push origin main"
echo ""
echo "2. Monitor Render deployment:"
echo "   - Check build logs in Render dashboard"
echo "   - Wait for deployment to complete (5-10 minutes)"
echo ""
echo "3. Test deployed service:"
echo "   python health_check.py https://medreserve-ml.onrender.com/health"
echo ""
echo "🔧 Troubleshooting:"
echo "==================="
echo "- If build fails: Use simple Dockerfile (run this script again)"
echo "- If startup fails: Check Render logs for specific errors"
echo "- If models fail: Service will use fallback predictions"
echo ""
echo "✅ Deployment fix completed!"
