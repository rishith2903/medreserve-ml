#!/bin/bash

# Setup script for MedReserve ML Repository
# This script helps you set up the ML service for deployment

echo "🤖 MedReserve ML Repository Setup"
echo "================================="

# Check if we're in the right directory
if [ ! -f "api/ml_api.py" ]; then
    echo "❌ Error: Please run this script from the backend/ml directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: api/ml_api.py"
    exit 1
fi

echo "✅ Found ML service files"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local

# Large model files (optional - comment out if you want to commit models)
# models/*.pkl

# Temporary files
tmp/
temp/
*.tmp
EOF
    echo "✅ .gitignore created"
fi

# Check if models exist
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "✅ Found trained models in models/ directory"
    echo "   Models will be included in the repository"
else
    echo "⚠️  No trained models found"
    echo "   Models will be trained during deployment (takes 2-5 minutes)"
fi

# Add all files to git
echo "📦 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit (repository is up to date)"
else
    echo "💾 Committing changes..."
    git commit -m "Add MedReserve ML service with Docker deployment configuration

- Add Dockerfile for containerized deployment
- Add render.yaml for Render platform deployment
- Add minimal requirements for faster builds
- Add startup script with automatic model training
- Add deployment documentation and setup scripts"
    echo "✅ Changes committed"
fi

# Check if remote origin exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "✅ Remote origin already configured"
    echo "   Origin: $(git remote get-url origin)"
else
    echo "⚠️  No remote origin configured"
    echo "   You'll need to add a remote origin before pushing:"
    echo "   git remote add origin https://github.com/rishith2903/medreserve-ml.git"
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. If you haven't created the GitHub repository yet:"
echo "   - Go to https://github.com/new"
echo "   - Create repository: medreserve-ml"
echo "   - Add remote: git remote add origin https://github.com/rishith2903/medreserve-ml.git"
echo ""
echo "2. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "3. Deploy to Render:"
echo "   - Go to https://dashboard.render.com"
echo "   - New Web Service"
echo "   - Connect medreserve-ml repository"
echo "   - Render will auto-detect Docker configuration"
echo ""
echo "4. Update main backend ML_SERVICE_URL:"
echo "   - Set to: https://medreserve-ml.onrender.com"
echo ""
echo "🚀 Your ML service is ready for deployment!"
