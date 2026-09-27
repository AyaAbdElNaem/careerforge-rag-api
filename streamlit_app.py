"""
streamlit_app.py — واجهة ومنطق CareerForge RAG مع بعض في ملف واحد.
يستخدم rag/query.py مباشرة (بدون FastAPI ولا Worker ولا Pages منفصلين).
"""

import streamlit as st
from rag.query import RagPipeline, query_rag

st.set_page_config(page_title="CareerForge RAG", page_icon="🎯", layout="centered")

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

st.title("🎯 CareerForge RAG")
st.caption("اسأل عن السيرة الذاتية، مقابلات الشغل، التفاوض على الراتب، أو مسارات التطور المهني في مجال البيانات.")


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


def ask(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والتفكير..."):
            result = query_rag(question, pipeline=pipeline)
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
