import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Load API Key ──────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# ── Custom CSS ────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
        /* ── Main background ── */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* ── Title ── */
        h1 {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }

        /* ── Subtitle ── */
        .subtitle {
            color: rgba(255,255,255,0.6);
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* ── Upload box ── */
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(167,139,250,0.5);
            border-radius: 12px;
            padding: 10px;
        }

        /* ── Process button ── */
        .stButton > button {
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.6rem 2rem;
            font-weight: 700;
            font-size: 1rem;
            width: 100%;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(167,139,250,0.4);
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.08);
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(167,139,250,0.4) !important;
            border-radius: 25px !important;
            color: white !important;
        }

        /* ── Success/warning boxes ── */
        .stSuccess {
            background: rgba(52,211,153,0.15) !important;
            border: 1px solid rgba(52,211,153,0.3) !important;
            border-radius: 10px !important;
        }

        .stWarning {
            background: rgba(251,191,36,0.15) !important;
            border: 1px solid rgba(251,191,36,0.3) !important;
            border-radius: 10px !important;
        }

        /* ── Expander ── */
        [data-testid="stExpander"] {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 10px !important;
        }

        /* ── PDF cards in sidebar ── */
        .pdf-card {
            background: rgba(167,139,250,0.1);
            border: 1px solid rgba(167,139,250,0.3);
            border-radius: 10px;
            padding: 8px 12px;
            margin: 5px 0;
            font-size: 0.85rem;
            color: rgba(255,255,255,0.9);
        }

        /* ── Stats cards ── */
        .stat-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            margin: 5px 0;
        }

        .stat-number {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* ── Language badge ── */
        .lang-badge {
            display: inline-block;
            background: rgba(96,165,250,0.2);
            border: 1px solid rgba(96,165,250,0.4);
            border-radius: 20px;
            padding: 2px 12px;
            font-size: 0.75rem;
            color: #60a5fa;
            margin-top: 5px;
        }

        /* ── Welcome banner ── */
        .welcome-banner {
            background: linear-gradient(135deg, 
                rgba(167,139,250,0.15), 
                rgba(96,165,250,0.15)
            );
            border: 1px solid rgba(167,139,250,0.3);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        }

        .welcome-banner h2 {
            color: white !important;
            font-size: 1.5rem !important;
        }

        .welcome-banner p {
            color: rgba(255,255,255,0.6);
            font-size: 0.95rem;
        }

        /* ── Sidebar header ── */
        .sidebar-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #a78bfa;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(167,139,250,0.3);
        }

        /* ── Hide streamlit branding ── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── STEP 1: Read PDF ──────────────────────────────
def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text
    return all_text


# ── STEP 2: Split text ────────────────────────────
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


# ── STEP 3: Create vector store ───────────────────
@st.cache_resource
def create_vector_store(texts_tuple, names_tuple):
    chunks = list(texts_tuple)
    names = list(names_tuple)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    metadatas = [{"source": name} for name in names]

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vector_store


# ── STEP 4: Get answer ────────────────────────────
def get_answer(vector_store, question, chat_history):
    docs = vector_store.similarity_search(question, k=6)

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[From: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    try:
        question_language = detect(question)
    except:
        question_language = "en"

    language_map = {
        "de": "German", "en": "English",
        "fr": "French", "es": "Spanish",
        "it": "Italian", "tr": "Turkish",
        "ar": "Arabic", "zh": "Chinese",
        "hi": "Hindi", "ja": "Japanese",
    }
    detected_language = language_map.get(question_language, "English")

    messages = []
    messages.append({
        "role": "system",
        "content": f"""You are a helpful multilingual assistant analyzing documents.
Each chunk is labeled [From: filename] so you know which document it came from.
Answer using ONLY the context below.
IMPORTANT: Always respond in {detected_language}!
If answer not in context, say in {detected_language}: "I don't know based on these documents."

Context:
{context}"""
    })

    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    return response.content, docs, detected_language


# ── STEP 5: UI ────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_css()

    # ── Session state ─────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_names" not in st.session_state:
        st.session_state.pdf_names = []
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "all_names" not in st.session_state:
        st.session_state.all_names = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0

    # ── SIDEBAR ───────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">📁 Upload your PDFs</div>',
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files and st.button("🚀 Process PDFs"):
            with st.spinner("Processing..."):
                all_chunks = []
                all_names = []

                for pdf_file in uploaded_files:
                    text = read_pdf(pdf_file)
                    chunks = split_text(text)
                    all_chunks.extend(chunks)
                    all_names.extend([pdf_file.name] * len(chunks))
                    st.write(f"✅ {pdf_file.name} → {len(chunks)} chunks")

                st.session_state.all_chunks = all_chunks
                st.session_state.all_names = all_names
                st.session_state.pdf_names = [f.name for f in uploaded_files]
                st.session_state.processed = True
                st.session_state.total_chunks = len(all_chunks)
                st.success(f"✅ {len(uploaded_files)} PDFs → {len(all_chunks)} chunks!")

        # ── Stats cards ───────────────────────────
        if st.session_state.processed:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(st.session_state.pdf_names)}</div>
                    <div class="stat-label">PDFs</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{st.session_state.total_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>""", unsafe_allow_html=True)

        # ── Uploaded PDF list ─────────────────────
        if st.session_state.pdf_names:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="sidebar-header">📂 Uploaded PDFs</div>',
                unsafe_allow_html=True
            )
            for name in st.session_state.pdf_names:
                st.markdown(
                    f'<div class="pdf-card">📄 {name}</div>',
                    unsafe_allow_html=True
                )

        # ── Languages ─────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        # ✅ New — using text instead of emoji
        st.markdown('<div style="font-size:13px; font-weight:500; color:#a78bfa; padding-bottom:8px; border-bottom:0.5px solid rgba(167,139,250,0.3);">🌍 Languages</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.85); font-size: 18px; letter-spacing: 4px; ">🇩🇪 🇬🇧 🇫🇷 🇪🇸 🇮🇹 🇹🇷 🇦🇪 🇨🇳 🇯🇵 🇮🇳</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.6); font-size: 12px; ">Supports 50+ languages including German & English</p>', unsafe_allow_html=True)

        # ── Clear button ──────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── MAIN AREA ─────────────────────────────────
    st.markdown("""
<h1>
    <span style="-webkit-text-fill-color: initial;">🌍</span>
    <span style="background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;">
     Multilingual PDF Chatbot
    </span>
</h1>
""", unsafe_allow_html=True)
    
    # Welcome banner when no PDFs uploaded
    if not st.session_state.processed:
        st.markdown("""
        <div class="welcome-banner">
            <h2>👋 Welcome! Let's chat with your PDFs</h2>
            <p>📁 Upload PDFs in the sidebar → 🚀 Click Process → 💬 Start asking!</p>
            <br>
            <p>✨ Supports <strong>50+ languages</strong> including German & English</p>
            <p>🧠 Remembers your <strong>conversation history</strong></p>
            <p>📄 Handles <strong>multiple PDFs</strong> at once</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat messages ─────────────────────────────
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # ── Chat input ────────────────────────────────
    question = st.chat_input(
        "Ask in any language... / Frag auf Deutsch... / Pregunta en español..."
    )

    if question:
        if not st.session_state.processed:
            st.warning("⚠️ Please upload and process PDFs first!")
        else:
            vector_store = create_vector_store(
                tuple(st.session_state.all_chunks),
                tuple(st.session_state.all_names)
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking... 🌍"):
                    answer, source_docs, detected_language = get_answer(
                        vector_store,
                        question,
                        st.session_state.chat_history
                    )
                    st.write(answer)

                    # Language badge
                    st.markdown(
                        f'<div class="lang-badge">🌍 {detected_language}</div>',
                        unsafe_allow_html=True
                    )

                    # Source chunks
                    with st.expander("📚 Source chunks used"):
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Chunk {i+1} — 📄 {source_name}**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()

            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

if __name__ == "__main__":
    main()