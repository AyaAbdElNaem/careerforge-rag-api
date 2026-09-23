FROM python:3.11-slim

WORKDIR /app

# نسخ ملف المتطلبات وتثبيته أولًا (لتسريع إعادة البناء لاحقًا)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي المشروع: فولدر rag/ وقاعدة chroma_db وملف app.py
COPY . .

# Hugging Face Spaces (Docker SDK) يتوقع السيرفر على المنفذ 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
