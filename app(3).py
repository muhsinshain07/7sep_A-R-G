import os

import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # open-source, small, fast (384 dims)
CHUNK_SIZE = 200      # words per chunk
CHUNK_OVERLAP = 40    # words shared between neighbouring chunks
TOP_K = 4             # chunks retrieved per question

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)


def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, step = [], max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def embed(texts, embedder) -> np.ndarray:
    vectors = np.array(list(embedder.embed(texts)), dtype="float32")
    faiss.normalize_L2(vectors)  # so inner product == cosine similarity
    return vectors


def build_index(chunks, embedder):
    vectors = embed(chunks, embedder)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query, index, chunks, embedder, k=TOP_K):
    q_vec = embed([query], embedder)
    scores, ids = index.search(q_vec, min(k, len(chunks)))
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def ask_groq(client, model, question, context_chunks, history):
    context = "\n\n---\n\n".join(c for c, _ in context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions about a PDF document. "
        "Use ONLY the context below. If the answer is not in the context, say "
        "you couldn't find it in the document.\n\n"
        f"CONTEXT:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-6:]  # a little conversation memory
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return key


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("file_name", None)
st.session_state.setdefault("messages", [])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")

    model = st.selectbox("Groq model (open-source)", GROQ_MODELS)

    st.divider()
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded and uploaded.name != st.session_state.file_name:
        with st.spinner("Reading PDF, creating embeddings, building FAISS index..."):
            text = extract_text(uploaded)
            if not text.strip():
                st.error("No text found. This may be a scanned PDF (image only).")
            else:
                chunks = chunk_text(text)
                embedder = load_embedder()
                st.session_state.index = build_index(chunks, embedder)
                st.session_state.chunks = chunks
                st.session_state.file_name = uploaded.name
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

    if st.session_state.file_name:
        st.caption(f"Active document: **{st.session_state.file_name}**")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("📄 Chat with your PDF")
st.caption("RAG app: pypdf → fastembed → FAISS → Groq (open-source LLMs)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask something about your PDF...")

if question:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()
    if st.session_state.index is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embedder = load_embedder()
                results = retrieve(
                    question, st.session_state.index, st.session_state.chunks, embedder
                )
                client = Groq(api_key=api_key)
                answer = ask_groq(
                    client, model, question, results, st.session_state.messages
                )
            except Exception as e:
                answer, results = f"Error: {e}", []
        st.markdown(answer)
        if results:
            with st.expander("Sources used"):
                for i, (chunk, score) in enumerate(results, 1):
                    st.markdown(f"**Chunk {i}** (similarity {score:.2f})")
                    st.write(chunk)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
