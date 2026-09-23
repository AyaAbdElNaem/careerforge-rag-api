"""
app.py — سيرفر بسيط بـ FastAPI يغلّف rag/query.py ويعرضه كـ endpoint واحد
POST /chat يستقبل {"question": "..."} ويرجّع {"answer": "...", "sources": [...]}.

يعمل على Hugging Face Spaces (Docker SDK) على المنفذ 7860.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.query import RagPipeline, query_rag

app = FastAPI(title="CareerForge RAG API")

# يسمح لأي واجهة (Cloudflare Pages/Worker) تنادي السيرفر ده
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# يتحمّل مرة واحدة فقط عند تشغيل السيرفر — مش مع كل طلب
pipeline = RagPipeline(persist_dir="./chroma_db")


class Question(BaseModel):
    question: str


@app.get("/")
def health():
    """endpoint بسيط للتأكد إن السيرفر شغال."""
    return {"status": "ok", "service": "CareerForge RAG"}


@app.post("/chat")
def chat(q: Question):
    return query_rag(q.question, pipeline=pipeline)
