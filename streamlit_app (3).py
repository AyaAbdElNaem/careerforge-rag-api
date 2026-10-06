"""
streamlit_app.py — واجهة ومنطق CareerForge RAG مع بعض في ملف واحد.
يستخدم rag/query.py مباشرة (بدون FastAPI ولا Worker ولا Pages منفصلين).
"""

import inspect

import streamlit as st
from rag.query import RagPipeline, query_rag

# اسم العرض والوصف وصورة الخلفية (رابط من النت — مفيش Base64 جوه الملف)
APP_DISPLAY_NAME = "DataPath AI"
APP_TAGLINE = "مساعدك الذكي للتطوير المهني في مجال البيانات"
BG_IMAGE_URL = "https://images.pexels.com/photos/27926566/pexels-photo-27926566.jpeg?auto=compress&cs=tinysrgb&h=627&fit=crop&w=1200"

st.set_page_config(page_title=APP_DISPLAY_NAME, page_icon="🎯", layout="centered")

# دعم اتجاه الكتابة من اليمين لليسار للنصوص العربية
st.markdown(
    """
    <style>
    .stChatMessage, .stMarkdown, .stTextInput, .stChatInput textarea {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# خلفية الصورة (من رابط) + طبقة تعتيم، مع ألوان نص فاتحة عشان تفضل مقروءة
st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(135deg, rgba(10,14,20,0.88) 0%, rgba(15,22,33,0.82) 45%, rgba(10,14,20,0.90) 100%),
            url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stBottom"] > div {{ background: transparent; }}
    .stApp h1, .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stChatMessage"] * {{ color: #f2f4f7; }}
    div.stButton > button {{
        background: rgba(15,20,28,0.5);
        border: 1px solid rgba(242,193,78,0.4);
    }}
    .stApp .hero {{ text-align: center; padding: 6px 0 18px; }}
    .stApp .hero .hero-title {{
        font-size: 2.7rem; font-weight: 800; letter-spacing: 0.5px; line-height: 1.2;
        direction: ltr; margin: 0;
        background: linear-gradient(90deg, #f2c14e, #e8973a);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .stApp .hero .hero-tagline {{
        direction: rtl; margin: 10px auto 0; max-width: 560px;
        font-size: 1.05rem; line-height: 1.7; color: #d7dee8;
    }}
    .stApp .hero .hero-line {{
        width: 64px; height: 3px; border-radius: 3px; margin: 14px auto 0;
        background: linear-gradient(90deg, #f2c14e, #e8973a);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# الهيدر: الاسم + الوصف في المنتصف (الـ HTML من غير مسافات بادئة وإلا Markdown يعتبره كود)
st.markdown(
    f"""<div class="hero">
<div class="hero-title">{APP_DISPLAY_NAME}</div>
<div class="hero-tagline">{APP_TAGLINE}</div>
<div class="hero-line"></div>
</div>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="جاري تحميل النماذج وقاعدة المعرفة...")
def load_pipeline() -> RagPipeline:
    """يتحمّل مرة واحدة بس ويتحفظ في الذاكرة بين الطلبات، بدل ما يتحمّل مع كل سؤال."""
    return RagPipeline(persist_dir="./chroma_db")


pipeline = load_pipeline()

# --- الأسئلة المقترحة ---
suggested = [
    "ما هو أسلوب STAR؟",
    "ما الأدوات التي يستخدمها مهندس البيانات؟",
    "كيف أستعد لمقابلة النظام (System Design)؟",
    "نصائح لكتابة سيرة ذاتية قوية",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 المصادر"):
        for s in sources:
            pages = s.get("pages", [])
            page_str = f"ص.{pages[0]}" if len(pages) == 1 else f"ص.{pages[0]}-{pages[-1]}"
            line = f"- **{s['source']}** ({page_str}"
            if s.get("section"):
                line += f"، {s['section']}"
            if s.get("table_id") is not None:
                line += f"، جدول {s['table_id']}"
            line += ")"
            st.markdown(line)


# لو query_rag بتدعم history هنبعتها، ولو لأ التطبيق يفضل شغّال زي الأول
_SUPPORTS_HISTORY = "history" in inspect.signature(query_rag).parameters


def ask(question: str):
    # سجل المحادثة كله من أول رسالة (قبل السؤال الحالي) — من غير اقتطاع
    history = [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources")}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والتفكير..."):
            extra = {"history": history} if (_SUPPORTS_HISTORY and history) else {}
            result = query_rag(question, pipeline=pipeline, **extra)
        st.write(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )


# --- عرض المحادثة السابقة ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))

# --- شيبس الأسئلة المقترحة (تظهر بس لو المحادثة لسه فاضية) ---
if not st.session_state.messages:
    st.write("جرّب واحد من الأسئلة دي:")
    cols = st.columns(2)
    for i, q in enumerate(suggested):
        if cols[i % 2].button(q, use_container_width=True):
            ask(q)
            st.rerun()

# --- صندوق الإدخال ---
if question := st.chat_input("اكتب سؤالك هنا..."):
    ask(question)
