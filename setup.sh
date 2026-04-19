#!/bin/bash
# setup.sh - تهيئة المشروع بسهولة

set -e  # التوقف عند أي خطأ

echo "═══════════════════════════════════════════════════════════"
echo "🤖 Company Document Chatbot - Setup"
echo "═══════════════════════════════════════════════════════════"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}🔍 فحص الإصدارات...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 إنشاء Virtual Environment...${NC}"
    python3 -m venv venv
    echo "✅ تم إنشاء venv"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ تم تفعيل venv"

# Upgrade pip
echo -e "${YELLOW}⬆️  تحديث pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo -e "${YELLOW}📥 تثبيت المتطلبات...${NC}"
pip install -r requirements.txt
echo "✅ تم تثبيت جميع المتطلبات"

# Create necessary directories
echo -e "${YELLOW}📁 إنشاء المجلدات...${NC}"
mkdir -p documents temp_uploads logs
echo "✅ تم إنشاء المجلدات"

# Copy .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚙️  نسخ ملف الإعدادات...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  تنبيه: ملأ OPENAI_API_KEY و PINECONE_API_KEY في .env${NC}"
else
    echo "✅ ملف .env موجود بالفعل"
fi

# Test imports
echo -e "${YELLOW}🧪 اختبار الاستيرادات...${NC}"
python3 -c "
import langchain
import pinecone
import fastapi
import streamlit
print('✅ جميع المكتبات تعمل بشكل صحيح')
" || {
    echo -e "${RED}❌ خطأ في الاستيراد${NC}"
    exit 1
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ تم الإعداد بنجاح!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "الخطوات التالية:"
echo "1. ملأ معلومات API في .env"
echo "2. ضع ملفات المستندات في مجلد ./documents"
echo ""
echo "للتشغيل:"
echo "  • Streamlit UI:  streamlit run app_streamlit.py"
echo "  • FastAPI:       uvicorn api_server:app --reload"
echo ""
echo "للـ Docker:"
echo "  docker-compose up -d"
echo ""
