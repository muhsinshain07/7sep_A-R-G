import os
from io import BytesIO

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# SETTINGS
# ============================================================
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 5


# ============================================================
# LOAD OPEN-SOURCE EMBEDDING MODEL
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# GROQ CLIENT
# ============================================================
@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================
def extract_pdf_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Clean unnecessary whitespace
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================
def create_chunks(pages):
    chunks = []

    for page_data in pages:
        words = page_data["text"].split()

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_data["page"],
                    }
                )

            if end >= len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================
def create_embeddings(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


# ============================================================
# VECTOR SEARCH USING NUMPY
#
# We use cosine similarity.
# Because embeddings are normalized, dot product = cosine similarity.
# This replaces FAISS and avoids the Streamlit Cloud FAISS error.
# ============================================================
def search_similar_chunks(
    question,
    chunks,
    embeddings,
    embedding_model,
    top_k=5,
):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")

    # Calculate similarity between the question and every chunk
    similarities = np.dot(embeddings, query_embedding)

    # Get the indexes of the most similar chunks
    top_k = min(top_k, len(chunks))

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "text": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": float(similarities[index]),
            }
        )

    return results


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================
def generate_answer(question, retrieved_chunks, groq_client):
    context = ""

    for item in retrieved_chunks:
        context += (
            f"\n[Page {item['page']}]\n"
            f"{item['text']}\n"
        )

    system_prompt = """
You are a helpful PDF question-answering assistant.

Use ONLY the information provided in the document context.

Rules:
1. Do not make up information.
2. If the answer cannot be found in the provided context, say:
   "I couldn't find the answer in the uploaded document."
3. Give a clear and concise answer.
4. Mention the relevant PDF page number when possible.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using only the document context.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================
st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content using "
    "open-source embeddings and Groq GPT-OSS."
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=2,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    st.markdown("### 🧠 Models")

    st.write(
        f"**Embedding:**\n"
        f"`{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:**\n"
        f"`{GROQ_MODEL}`"
    )

    st.write("**Vector Search:** NumPy Cosine Similarity")


# ============================================================
# PDF UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"],
)


if uploaded_file is None:

    st.info("👆 Upload a PDF file to start.")

    st.markdown(
        """
        ### How this application works

        1. Upload a PDF
        2. Extract text
        3. Split text into chunks
        4. Create open-source embeddings
        5. Search for relevant chunks
        6. Send relevant context to Groq
        7. Generate an answer
        """
    )

else:

    # ========================================================
    # RESET DATA WHEN A NEW PDF IS UPLOADED
    # ========================================================
    if (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    ):
        st.session_state.file_name = uploaded_file.name
        st.session_state.pdf_bytes = uploaded_file.getvalue()

        st.session_state.pages = None
        st.session_state.chunks = None
        st.session_state.embeddings = None

    # ========================================================
    # EXTRACT PDF
    # ========================================================
    if st.session_state.pages is None:

        with st.spinner("📖 Reading the PDF..."):

            st.session_state.pages = extract_pdf_text(
                st.session_state.pdf_bytes
            )

    if not st.session_state.pages:

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.warning(
            "This may be a scanned/image-only PDF. "
            "OCR will be needed for scanned PDFs."
        )

        st.stop()

    # ========================================================
    # CREATE CHUNKS
    # ========================================================
    if st.session_state.chunks is None:

        with st.spinner("✂️ Splitting the PDF into chunks..."):

            st.session_state.chunks = create_chunks(
                st.session_state.pages
            )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================
    with st.spinner(
        "🧠 Loading the open-source embedding model..."
    ):

        embedding_model = load_embedding_model()

    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================
    if st.session_state.embeddings is None:

        with st.spinner(
            "🔢 Creating document embeddings..."
        ):

            st.session_state.embeddings = create_embeddings(
                st.session_state.chunks,
                embedding_model,
            )

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF Pages",
        len(st.session_state.pages),
    )

    col2.metric(
        "Text Chunks",
        len(st.session_state.chunks),
    )

    col3.metric(
        "Embedding Size",
        st.session_state.embeddings.shape[1],
    )

    st.success(
        "✅ PDF processed successfully. "
        "You can now ask questions."
    )

    # ========================================================
    # QUESTION INPUT
    # ========================================================
    question = st.text_input(
        "💬 Ask a question about your PDF",
        placeholder="Example: What is the main topic of this document?",
    )

    if question:

        # ====================================================
        # LOAD GROQ
        # ====================================================
        groq_client = load_groq_client()

        if groq_client is None:

            st.error(
                "❌ GROQ_API_KEY is missing."
            )

            st.info(
                "For Streamlit Cloud, add GROQ_API_KEY "
                "under your app's Secrets."
            )

            st.stop()

        # ====================================================
        # RETRIEVAL
        # ====================================================
        with st.spinner(
            "🔎 Searching the document..."
        ):

            retrieved_chunks = search_similar_chunks(
                question=question,
                chunks=st.session_state.chunks,
                embeddings=st.session_state.embeddings,
                embedding_model=embedding_model,
                top_k=top_k,
            )

        # ====================================================
        # GENERATION
        # ====================================================
        with st.spinner(
            "🤖 Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    groq_client,
                )

                st.subheader("🤖 Answer")

                st.write(answer)

            except Exception as error:

                st.error(
                    f"❌ Groq API error: {error}"
                )

        # ====================================================
        # SHOW RETRIEVED CHUNKS
        # ====================================================
        with st.expander(
            "🔎 View retrieved document chunks"
        ):

            for number, item in enumerate(
                retrieved_chunks,
                start=1,
            ):

                st.markdown(
                    f"### Chunk {number}"
                )

                st.write(
                    f"**PDF Page:** {item['page']}"
                )

                st.write(
                    f"**Similarity:** "
                    f"{item['score']:.3f}"
                )

                st.write(item["text"])

                st.divider()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "RAG Pipeline: PDF → Text → Chunks → "
    "Open-Source Embeddings → Vector Search → "
    "Groq GPT-OSS → Answer"
)
