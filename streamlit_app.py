"""
IlmGPT — Islamic Scholar Assistant
Streamlit Web Application
Run with: streamlit run streamlit_app.py
"""

import os
import sys
import re
import textwrap
import html
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from groq import Groq


def get_saved_api_key(key_name: str = "GOOGLE_API_KEY") -> str:
    """Load an API key from environment or Streamlit secrets."""
    key = os.environ.get(key_name, "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get(key_name, "")).strip()
    except Exception:
        return ""

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "IlmGPT — Islamic Scholar Assistant",
    page_icon  = "🕌",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ── Premium 3D Glassmorphic Islamic Theme ──────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Tajawal:wght@300;400;500;700&family=Cinzel:wght@400;600;700&display=swap');

    :root {
        --gold: #d4af37;
        --gold-light: #f0d060;
        --gold-dim: rgba(212,175,55,0.25);
        --navy-deep: #050d1a;
        --text-primary: #eef4ff;
        --text-secondary: rgba(238,244,255,0.65);
        --border-glass: rgba(212,175,55,0.18);
        --radius: 14px;
    }

    /* Deep Islamic night background */
    .stApp {
        background:
            radial-gradient(ellipse at 20% 10%, rgba(30,15,60,0.8) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 90%, rgba(10,30,70,0.9) 0%, transparent 55%),
            radial-gradient(ellipse at 50% 50%, rgba(5,15,35,1) 0%, #020810 100%);
        min-height: 100vh;
    }

    /* Geometric dot pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle at 1px 1px, rgba(212,175,55,0.06) 1px, transparent 0);
        background-size: 38px 38px;
        pointer-events: none;
        z-index: 0;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 960px;
        position: relative;
        z-index: 1;
    }

    /* ── Header ── */
    .ilm-header {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        margin-bottom: 2rem;
    }
    .ilm-header::after {
        content: '';
        display: block;
        width: 300px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        margin: 1.2rem auto 0;
    }
    .ilm-arch {
        color: rgba(212,175,55,0.3);
        letter-spacing: 16px;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .bismillah {
        font-family: 'Amiri', serif;
        font-size: 2.2rem;
        color: var(--gold);
        text-shadow: 0 0 20px rgba(212,175,55,0.5), 0 0 60px rgba(212,175,55,0.15);
        display: block;
        margin-bottom: 0.8rem;
    }
    .ilm-title {
        font-family: 'Cinzel', serif;
        font-size: 3.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #b8962e 70%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: 4px;
        filter: drop-shadow(0 2px 12px rgba(212,175,55,0.4));
    }
    .ilm-subtitle {
        font-family: 'Tajawal', sans-serif;
        color: var(--text-secondary);
        font-size: 0.88rem;
        font-weight: 300;
        margin-top: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* ── Answer card with 3D glass effect ── */
    .answer-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.055) 0%, rgba(10,22,45,0.6) 100%);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        padding: 2.5rem;
        margin: 1.2rem 0;
        font-family: 'Tajawal', sans-serif;
        color: var(--text-primary);
        line-height: 1.9;
        box-shadow:
            0 20px 60px rgba(0,0,0,0.5),
            0 4px 16px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    .answer-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        opacity: 0.6;
    }

    .tab-header {
        font-family: 'Cinzel', serif;
        color: var(--gold);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--gold-dim);
        padding-bottom: 0.5rem;
    }

    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 1.08rem;
        line-height: 2.1;
        font-family: 'Tajawal', sans-serif;
    }

    .urdu-text.tab-header {
        font-family: 'Amiri', serif;
        font-size: 1.4rem;
    }

    .gold-circle {
        background: linear-gradient(135deg, #d4af37 0%, #b8962e 100%);
        color: #000;
        width: 24px;
        height: 24px;
        min-width: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    .citation-badge {
        background: rgba(212,175,55,0.2);
        color: var(--gold-light);
        border: 1px solid rgba(212,175,55,0.4);
        padding: 0.1rem 0.4rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 0.3rem;
        vertical-align: middle;
    }

    .summary-block {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid var(--gold);
        padding: 1rem 1.5rem;
        margin-top: 1.5rem;
        border-radius: 0 8px 8px 0;
        color: var(--text-secondary);
    }

    .urdu-text .summary-block {
        border-left: none;
        border-right: 3px solid var(--gold);
        border-radius: 8px 0 0 8px;
    }

    .sources-strip {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-glass);
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
    }

    .source-chip {
        background: rgba(0,0,0,0.4);
        border: 1px solid var(--border-glass);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.8);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .source-chip .citation-badge {
        margin: 0;
        padding: 0 0.4rem;
        border-radius: 4px;
        background: var(--gold);
        color: #000;
    }

    /* ── Inputs — warm parchment ── */
    .stTextArea textarea,
    .stTextInput input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="base-input"] input {
        background: #faf6e9 !important;
        border: 1px solid rgba(212,175,55,0.4) !important;
        color: #1a1208 !important;
        -webkit-text-fill-color: #1a1208 !important;
        caret-color: #8b6914 !important;
        border-radius: 10px !important;
        font-family: 'Tajawal', sans-serif !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3), inset 0 2px 4px rgba(0,0,0,0.08) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(212,175,55,0.2), 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(26,18,8,0.45) !important;
        -webkit-text-fill-color: rgba(26,18,8,0.45) !important;
        font-style: italic;
    }

    /* Body text */
    .main, .main p, .main li, .main span, .main div,
    [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
        font-family: 'Tajawal', sans-serif;
    }
    [data-testid="stMarkdownContainer"] strong { color: #f0d060 !important; }

    /* ── Ask button — gold 3D ── */
    .stButton button {
        background: linear-gradient(135deg, #d4af37 0%, #c9a227 50%, #b8962e 100%) !important;
        color: #0d0a02 !important;
        font-weight: 700 !important;
        font-family: 'Cinzel', serif !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.45rem 1.5rem !important;
        font-size: 0.95rem !important;
        letter-spacing: 1px !important;
        box-shadow:
            0 4px 20px rgba(212,175,55,0.35),
            inset 0 1px 0 rgba(255,255,255,0.25),
            inset 0 -2px 0 rgba(0,0,0,0.15) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(212,175,55,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
        background: linear-gradient(135deg, #e0bb42 0%, #d4af37 50%, #c9a227 100%) !important;
    }
    .stButton button:active { transform: translateY(0px) !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5,12,28,0.98) 0%, rgba(8,18,40,0.98) 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--gold) !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.88rem !important;
        letter-spacing: 1.5px !important;
    }

    /* Sidebar example buttons */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, rgba(212,175,55,0.1) 0%, rgba(212,175,55,0.04) 100%) !important;
        color: rgba(255,255,255,0.75) !important;
        border: 1px solid rgba(212,175,55,0.2) !important;
        font-family: 'Tajawal', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        letter-spacing: 0 !important;
        padding: 0.45rem 0.8rem !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, rgba(212,175,55,0.2) 0%, rgba(212,175,55,0.08) 100%) !important;
        border-color: rgba(212,175,55,0.4) !important;
        color: #f0d060 !important;
        transform: translateX(3px) !important;
        box-shadow: none !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(10,22,45,0.5) 100%) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(212,175,55,0.35) !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stExpander"] summary { color: var(--gold) !important; }

    /* Section labels */
    .section-label {
        font-family: 'Cinzel', serif;
        font-size: 0.82rem;
        color: var(--gold);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--gold-dim), transparent);
    }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--gold-dim), transparent) !important;
    }
    .stSpinner > div { border-top-color: var(--gold) !important; }

    .ilm-footer {
        text-align: center;
        padding: 2rem 1rem 1rem;
        border-top: 1px solid var(--border-glass);
        margin-top: 3rem;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--navy-deep); }
    ::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    db_path = "../db/chroma" if os.path.exists("../db/chroma") else "db/chroma"
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chroma_client   = chromadb.PersistentClient(path=db_path)
    collection      = chroma_client.get_collection("ilmgpt_collection")
    return embedding_model, collection


@st.cache_resource
def load_llm(api_key: str, provider: str = "Groq"):
    if provider == "Gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.05, max_tokens=1024)
    else:
        os.environ["GROQ_API_KEY"] = api_key
        return Groq(api_key=api_key)


# ── Core RAG Functions ────────────────────────────────────────────────────────
def retrieve(query, embedding_model, collection, top_k=5, source_filter=None):
    normalized = query.lower().strip()
    topic_expansions = {
        "salah": "salah salat prayer virtues",
        "salat": "salah salat prayer virtues",
        "prayer": "salah salat prayer namaz virtues",
        "namaz": "salah salat prayer namaz نماز الصلاة",
        "نماز": "salah salat prayer namaz نماز الصلاة",
        "الصلاة": "salah salat prayer namaz نماز الصلاة",
        "forgiveness": "forgiveness repentance tawbah maghfirah",
        "patience": "patience sabr صبر",
        "charity": "charity sadaqah zakat صدقة زكاة",
        "honesty": "honesty truthfulness sidq صدق صادق",
        "lying": "lying falsehood kadhib كذب dishonesty",
    }
    expansion_terms = [v for k, v in topic_expansions.items() if k in normalized]
    expanded_query = f"{query} {' '.join(expansion_terms)}" if expansion_terms else query

    query_vector = embedding_model.encode([expanded_query]).tolist()
    where_filter = {"type": source_filter} if source_filter else None
    return collection.query(query_embeddings=query_vector, n_results=top_k, where=where_filter)


def build_prompt(query, results):
    """Build a strict, grounded anti-hallucination RAG prompt."""
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        ref = meta.get("reference", "Unknown Source")
        context_parts.append(f"[{i+1}] SOURCE — {ref}:\n{doc}")
    context = "\n\n".join(context_parts)

    return f"""You are IlmGPT, a respectful Islamic Scholar Assistant. Your ONLY job is to report what the SOURCES SAY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT ANTI-HALLUCINATION RULES — FOLLOW EXACTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ONLY use information directly stated in the numbered sources below. Add NOTHING from your own training.
2. NEVER present a paraphrase as a direct quote. Describe what the source says.
3. Do NOT merge content from two different hadiths into one statement.
4. Do NOT add fiqh rulings, extra context, or scholarly opinion not found in the sources.
5. If sources do not answer the question, say exactly: "The provided sources do not contain a direct answer. Please consult a qualified scholar."
6. Use the marker SOURCE_n to cite a source, where n is the source number. Example: "The Prophet said... SOURCE_1". Do NOT use [1].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROVIDED SOURCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {query}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Answer in EXACTLY this format (do not use prefixes like "Point 1:", just start with the point):

**English Answer:**
- [What the source actually states] SOURCE_n
- [What the source actually states] SOURCE_n

**English Summary:**
[One sentence — only based on provided sources]

**اردو جواب:**
- [ماخذ کے مطابق] SOURCE_n
- [ماخذ کے مطابق] SOURCE_n

**اردو خلاصہ:**
[ایک جملہ — صرف ماخذ کی بنیاد پر]
"""


def ask_ilmgpt(question, llm, embedding_model, collection, top_k=5, source_filter=None, provider="Groq"):
    results = retrieve(question, embedding_model, collection, top_k, source_filter)
    prompt  = build_prompt(question, results)
    try:
        if provider == "Groq":
            resp = llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.05,
                max_tokens=1024,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict Islamic source-reporting assistant. "
                            "Never fabricate hadith or Quranic quotes. "
                            "Never add information beyond what is in the provided numbered sources. "
                            "Describe what each source says — do not present paraphrases as direct quotes. "
                            "Always use SOURCE_n for citations."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
            )
            return resp.choices[0].message.content, results

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, results

    except Exception as exc:
        error_text = str(exc)
        if "403" in error_text or "denied" in error_text.lower():
            msg = "Gemini access denied for this project."
        elif "429" in error_text or "quota" in error_text.lower():
            msg = "Gemini rate-limited."
        else:
            msg = "LLM currently unavailable."

        fallback = (
            "**English Answer:**\n"
            f"- {msg} Showing retrieved sources only. SOURCE_1\n"
            "**English Summary:**\nLLM unavailable.\n"
            "**اردو جواب:**\n"
            "- LLM ابھی دستیاب نہیں۔ نیچے متعلقہ مصادر ملاحظہ فرمائیں۔ SOURCE_1\n"
            "**اردو خلاصہ:**\nLLM دستیاب نہیں۔\n"
        )
        st.warning(f"LLM unavailable: {exc}")
        return fallback, results


def render_answer(answer_text, results):
    eng_ans, eng_sum = "", ""
    urd_ans, urd_sum = "", ""
    current_section = None
    
    for line in answer_text.split('\n'):
        line_stripped = line.strip()
        if "**English Answer:**" in line: current_section = "eng_ans"; continue
        elif "**English Summary:**" in line: current_section = "eng_sum"; continue
        elif "**اردو جواب:**" in line: current_section = "urd_ans"; continue
        elif "**اردو خلاصہ:**" in line: current_section = "urd_sum"; continue
        elif "Sources Used:" in line: continue
            
        if current_section == "eng_ans" and line_stripped: eng_ans += line + "\n"
        elif current_section == "eng_sum" and line_stripped: eng_sum += line + "\n"
        elif current_section == "urd_ans" and line_stripped: urd_ans += line + "\n"
        elif current_section == "urd_sum" and line_stripped: urd_sum += line + "\n"

    def format_bullets(text, is_urdu=False):
        rendered = ""
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            if line.startswith('-'): line = line[1:].strip()
            line = html.escape(line)
            line = re.sub(r'SOURCE_(\d+)', r'<span class="citation-badge">\1</span>', line)
            dir_style = 'direction: rtl;' if is_urdu else ''
            rendered += (
                f'<div style="display: flex; gap: 0.8rem; margin-bottom: 0.8rem; {dir_style}">' 
                f'<div class="gold-circle">{i+1}</div>'
                f'<div>{line}</div>'
                '</div>'
            )
        return rendered

    eng_bullets = format_bullets(eng_ans)
    urd_bullets = format_bullets(urd_ans, is_urdu=True)

    sources_html = ""
    for i, meta in enumerate(results["metadatas"][0]):
        ref = meta.get("reference", "Unknown")
        ref_short = ref.split(" - ")[0] if " - " in ref else ref
        sources_html += f'<span class="source-chip"><span class="citation-badge">{i+1}</span> {ref_short}</span>'

    eng_summary = html.escape(eng_sum.strip())
    urd_summary = html.escape(urd_sum.strip())

    return (
        '<div class="answer-card">'
        '<div class="tab-header">🇬🇧 English Answer</div>'
        '<div class="tab-content">'
        f'{eng_bullets}'
        f'<div class="summary-block"><i>{eng_summary}</i></div>'
        '</div>'
        '<hr style="margin: 2rem 0; opacity: 0.2;">'
        '<div class="tab-header urdu-text">🕌 اردو جواب</div>'
        '<div class="tab-content urdu-text">'
        f'{urd_bullets}'
        f'<div class="summary-block"><i>{urd_summary}</i></div>'
        '</div>'
        '<div class="sources-strip">'
        f'{sources_html}'
        '</div>'
        '</div>'
    )

# ── UI Layout ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ilm-header">
    <div class="ilm-arch">✦ &nbsp; &nbsp; ✦ &nbsp; &nbsp; ✦</div>
    <span class="bismillah">بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</span>
    <h1 class="ilm-title">IlmGPT</h1>
    <p class="ilm-subtitle">Islamic Scholar Assistant &nbsp;·&nbsp; Quran &amp; Hadith Q&amp;A with Sources</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    llm_provider  = st.selectbox("LLM Provider", ["Groq (Free & Fast)", "Gemini"], index=0)
    provider_name = "Groq" if "Groq" in llm_provider else "Gemini"

    if provider_name == "Gemini":
        saved = get_saved_api_key("GOOGLE_API_KEY")
        api_key = st.text_input("Google Gemini API Key", type="password",
                                help="aistudio.google.com",
                                placeholder="Loaded automatically" if saved else "AIza...",
                                value=saved)
        if saved: st.caption("✅ Gemini key loaded from secrets.")
    else:
        saved = get_saved_api_key("GROQ_API_KEY")
        api_key = st.text_input("Groq API Key", type="password",
                                help="console.groq.com",
                                placeholder="Loaded automatically" if saved else "gsk_...",
                                value=saved)
        if saved: st.caption("✅ Groq key loaded from secrets.")

    st.markdown("---")
    source_option = st.selectbox("Search In", ["Both (Quran + Hadith)", "Quran Only", "Hadith Only"])
    top_k = st.slider("Sources to Retrieve", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 📖 Example Questions")
    examples = [
        "What does Islam say about patience?",
        "How should Muslims treat their parents?",
        "What is the importance of Salah?",
        "What did Prophet ﷺ say about honesty?",
        "نماز کی فضیلت کیا ہے؟",
        "What does Quran say about forgiveness?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question_input"] = ex
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.35); font-size:0.74rem; line-height:1.9;'>
    Built by <b style='color:#d4af37'>Muhammad Wasif</b><br>
    LangChain · ChromaDB · Gemini · Groq<br>
    Dataset: Quran + Sahih Hadith
    </div>""", unsafe_allow_html=True)


# ── Main Input ──
st.markdown('<div class="section-label">🔍 &nbsp; Ask a Question</div>', unsafe_allow_html=True)

question = st.text_area(
    "question_area",
    value       = st.session_state.get("question_input", ""),
    height      = 110,
    placeholder = "e.g. What does Islam say about kindness to parents?",
    key         = "main_question",
    label_visibility = "collapsed"
)

col1, col2, col3 = st.columns([3.5, 2, 3.5])
with col2:
    ask_button = st.button("🔍  Ask IlmGPT", key="ask_submit", use_container_width=True)


# ── Run Pipeline ──
if ask_button:
    if not api_key:
        st.error(f"❌ Please enter your {provider_name} API key in the sidebar.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        try:
            filter_map = {
                "Both (Quran + Hadith)": None,
                "Quran Only": "quran",
                "Hadith Only": "hadith"
            }
            source_filter = filter_map[source_option]

            with st.spinner("Loading Islamic knowledge base..."):
                embedding_model, collection = load_resources()
                llm = load_llm(api_key, provider_name)

            with st.spinner(f"Searching Quran & Hadith · Consulting {provider_name}..."):
                answer, results = ask_ilmgpt(
                    question, llm, embedding_model, collection,
                    top_k=top_k, source_filter=source_filter, provider=provider_name
                )

            # Display answer
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">📝 &nbsp; Answer</div>', unsafe_allow_html=True)
            # Render inside glass card; use markdown for bold/bullets
            html_answer = render_answer(answer, results)
            st.markdown(html_answer, unsafe_allow_html=True)

            # Display sources
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">📚 &nbsp; Retrieved Sources</div>', unsafe_allow_html=True)

            num_cols = min(len(results["documents"][0]), 3)
            cols = st.columns(num_cols)
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                with cols[i % num_cols]:
                    src_type = meta.get("type", "").upper()
                    ref      = meta.get("reference", "Unknown")
                    icon     = "📖" if src_type == "QURAN" else "📜"
                    with st.expander(f"{icon} {ref}"):
                        st.write(doc[:500] + ("…" if len(doc) > 500 else ""))

        except Exception as e:
            if "not found" in str(e).lower() or "collection" in str(e).lower():
                st.error("❌ Database not found. Please run the notebook first to build the ChromaDB index.")
            else:
                st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure you ran all cells in `ilmgpt_notebook.ipynb` first.")


# ── Footer ──
st.markdown("""
<div class="ilm-footer">
    <div style='color:rgba(212,175,55,0.4); font-size:0.8rem; font-family:Tajawal,sans-serif; line-height:2;'>
        ⚠️ IlmGPT is an educational tool. Always verify with a qualified Islamic scholar for important matters.<br>
        <span style='font-size:0.85rem;'>یہ ایک تعلیمی ٹول ہے۔ اہم معاملات میں کسی اہل عالم سے رجوع کریں۔</span>
    </div>
</div>
""", unsafe_allow_html=True)
