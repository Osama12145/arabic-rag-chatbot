# 🚀 Deployment Guide - دليل النشر الكامل

## الخيارات المتاحة للنشر

---

## **Option 1: نشر محلي (Local - للتطوير)**

### متطلبات:
- Python 3.10+
- 4GB RAM minimum
- Pinecone Account (مجاني)
- OpenAI API Key

### الخطوات:

```bash
# 1. استنساخ وتهيئة
bash setup.sh

# 2. ملأ .env
nano .env
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=...

# 3. تشغيل في terminals منفصلة:

# Terminal 1: Streamlit UI
streamlit run app_streamlit.py

# Terminal 2: FastAPI Server (اختياري)
uvicorn api_server:app --reload --port 8000
```

**الوصول:**
- UI: http://localhost:8501
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## **Option 2: Docker (اموصى به للـ Production)**

### متطلبات:
- Docker & Docker Compose
- 8GB RAM
- Pinecone + OpenAI accounts

### الخطوات:

```bash
# 1. ملأ .env
cp .env.example .env
# ملأ API keys

# 2. بناء وتشغيل
docker-compose up -d

# 3. فحص الحالة
docker-compose ps
docker-compose logs -f

# 4. التوقف
docker-compose down
```

**الخدمات:**
- API: http://localhost:8000
- UI: http://localhost:8501
- DB: localhost:5432
- Cache: localhost:6379

---

## **Option 3: AWS EC2**

### المتطلبات:
- EC2 Instance (t3.medium أو أعلى)
- Ubuntu 22.04 LTS
- Security Group مفتوح

### خطوات النشر:

#### 1. تجهيز الـ Instance
```bash
# SSH إلى الـ instance
ssh -i your-key.pem ubuntu@your-instance-ip

# تحديث النظام
sudo apt update && sudo apt upgrade -y

# تثبيت الأساسيات
sudo apt install -y python3.11 python3-pip docker.io docker-compose git

# إضافة مستخدم docker
sudo usermod -aG docker ubuntu
```

#### 2. استنساخ المشروع
```bash
cd /home/ubuntu
git clone https://github.com/yourusername/company-chatbot.git
cd company-chatbot

# ملأ الإعدادات
cp .env.example .env
nano .env  # أضف API keys
```

#### 3. النشر
```bash
# استخدم Docker
docker-compose up -d

# أو شغل مباشرة
bash setup.sh
streamlit run app_streamlit.py &
uvicorn api_server:app --host 0.0.0.0 --port 8000 &
```

#### 4. إعداد Nginx Reverse Proxy
```bash
# تثبيت Nginx
sudo apt install nginx

# نسخ الإعدادات
sudo tee /etc/nginx/sites-available/chatbot > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;

    # API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # Streamlit
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# تفعيل
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. إعداد SSL (HTTPS)
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## **Option 4: Heroku**

### خطوات النشر:

#### 1. الإعداد
```bash
# تثبيت Heroku CLI
curl https://cli.heroku.com/install.sh | sh

# تسجيل الدخول
heroku login

# إنشاء تطبيق
heroku create your-chatbot-name
```

#### 2. إنشاء Procfile
```procfile
web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
worker: streamlit run app_streamlit.py
```

#### 3. إضافة Config Variables
```bash
heroku config:set OPENAI_API_KEY=sk-...
heroku config:set PINECONE_API_KEY=...
heroku config:set DATABASE_URL=...
```

#### 4. النشر
```bash
git push heroku main
heroku logs --tail
```

---

## **Option 5: DigitalOcean App Platform**

### خطوات النشر:

#### 1. إنشاء Repository
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

#### 2. ربط مع DigitalOcean
- اذهب إلى DigitalOcean Dashboard
- اضغط "Create" → "App"
- اختر GitHub repository
- اضغط "Next"

#### 3. الإعدادات
```yaml
# في DigitalOcean App Spec
services:
  - name: api
    github:
      repo: yourusername/company-chatbot
      branch: main
    build_command: pip install -r requirements.txt
    run_command: uvicorn api_server:app
    http_port: 8000
    envs:
      - key: OPENAI_API_KEY
        scope: RUN_AND_BUILD_TIME
        value: ${OPENAI_API_KEY}
```

---

## **Option 6: Railway.app (الأسهل)**

### خطوات النشر:

```bash
# 1. تثبيت Railway CLI
npm i -g @railway/cli

# 2. الدخول
railway login

# 3. إنشاء مشروع
railway init

# 4. إضافة متغيرات البيئة
railway variables set OPENAI_API_KEY=sk-...

# 5. النشر
railway up

# 6. الوصول
railway open
```

---

## **Performance Optimization**

### Caching
```python
# في api_server.py
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query):
    # ...
```

### Database Indexing
```sql
-- في PostgreSQL
CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_embeddings_metadata ON embeddings USING GIN(metadata);
```

### Load Balancing
```bash
# استخدم Nginx
upstream api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}
```

---

## **Monitoring & Logging**

### استخدام ELK Stack
```docker
version: '3'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
    environment:
      - discovery.type=single-node
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.0.0
    ports:
      - "5601:5601"
```

### استخدام Sentry
```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0
)
```

---

## **Database Backups**

```bash
# Backup PostgreSQL
pg_dump chatbot_db > backup.sql

# Restore
psql chatbot_db < backup.sql

# Pinecone Backup (في الكود)
export_documents = vs_manager.vectorstore.similarity_search_with_scores("*", k=10000)
```

---

## **الـ Checklist النهائي**

- [ ] API keys محفوظة بشكل آمن
- [ ] Database مدعومة بانتظام
- [ ] SSL/HTTPS مفعل
- [ ] Logging و Monitoring مفعل
- [ ] Rate limiting مفعل
- [ ] Error handling كامل
- [ ] Documentation محدثة
- [ ] Tests مُمررة
- [ ] Performance optimized
- [ ] Security verified

---

## **استكشاف الأخطاء**

### Memory Issues
```bash
# زد الـ swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Timeout Issues
```python
# في FastAPI
from fastapi import BackgroundTasks
@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_async, request.query)
```

### Database Connection Pool
```python
# في SQLAlchemy
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)
```

---

## **التكاليف المتوقعة**

| الخدمة | الخطة | السعر الشهري |
|--------|------|-----------|
| OpenAI API | Pay-as-you-go | $5-50 |
| Pinecone | Starter | مجاني (1GB) |
| AWS EC2 | t3.medium | $30-50 |
| PostgreSQL | RDS | $15-30 |
| Monitoring | DataDog | $15-100 |

**الإجمالي الأدنى:** ~$100/شهر

---

## **النصائح النهائية**

1. **ابدأ محلياً** ثم انتقل للـ Production
2. **استخدم CI/CD** مع GitHub Actions
3. **راقب الأداء** باستمرار
4. **اعمل backup** منتظم
5. **حدث المتطلبات** دورياً
6. **اختبر قبل النشر** (testing!)

---

**تم إعداد هذا الدليل لـ 2024 - قد تتغير بعض الأسعار والإصدارات**
