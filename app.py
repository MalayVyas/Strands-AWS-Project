import os
import io
import asyncio
import math
import streamlit as st
from dotenv import load_dotenv

# Strands
from strands import Agent
from strands.models.litellm import LiteLLMModel

# PDF + retrieval
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Strands × Gemini PDF QA", page_icon="📄", layout="centered")
load_dotenv()


# -----------------------------
# Key resolution
# -----------------------------
def resolve_gemini_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if not key:
            key = st.secrets.get("GOOGLE_API_KEY", None)
    except FileNotFoundError:
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")


GEMINI_API_KEY = resolve_gemini_key()


# -----------------------------
# Sidebar: settings + PDF upload
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add to .env or .streamlit/secrets.toml.")
    model_id = st.selectbox(
        "Gemini model (LiteLLM format)",
        [
            os.getenv("MODEL_ID", "gemini/gemini-2.0-flash"),
            "gemini/gemini-2.5-pro",
            "gemini/gemini-1.5-flash",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("TEMPERATURE", "0.3")), 0.05)
    sys_prompt = st.text_area(
        "System prompt",
        value="You answer using ONLY the provided PDF excerpts. If unsure, reply exactly: not sure.",
        height=120,
    )

    st.markdown("---")
    st.subheader("📄 Load a PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

    st.caption("Tip: larger PDFs may take a few seconds to index on first load.")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Clear chat"):
            st.session_state.pop("messages", None)
            st.toast("Chat cleared.")
    with colB:
        if st.button("♻️ Clear PDF index"):
            for k in ("pdf_chunks", "pdf_vectorizer", "pdf_matrix", "pdf_name"):
                st.session_state.pop(k, None)
            st.toast("PDF index cleared.")


st.title("📄 Strands × Gemini — Ask your PDF")


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]


# -----------------------------
# PDF helpers
# -----------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF (very simple; no layout reconstruction)."""
    txt = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        # 'extract_text' returns a string, sometimes with newlines
        txt.append(page.extract_text() or "")
    return "\n".join(txt)


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> list[str]:
    """
    Dumb but effective chunking by characters, with overlap.
    Keeps context near boundaries and avoids extra deps.
    """
    text = text.replace("\r", "")
    n = len(text)
    chunks = []
    i = 0
    while i < n:
        chunk = text[i : i + chunk_chars]
        chunks.append(chunk.strip())
        i += chunk_chars - overlap
        if i < 0 or i >= n:
            break
    # Filter very short chunks
    return [c for c in chunks if len(c) > 20]


def ensure_pdf_index(file, force=False):
    """
    Build TF-IDF index for the uploaded PDF and cache it in session state.
    """
    if not file:
        return

    name_changed = st.session_state.get("pdf_name") != file.name
    if "pdf_chunks" in st.session_state and not (force or name_changed):
        return  # already indexed this file

    # Read file and build chunks
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    chunks = chunk_text(text)

    if not chunks:
        st.warning("Could not extract any text from the PDF.")
        return

    # Build TF-IDF vectors for chunks
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        max_features=50_000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(chunks)

    st.session_state.pdf_chunks = chunks
    st.session_state.pdf_vectorizer = vectorizer
    st.session_state.pdf_matrix = matrix
    st.session_state.pdf_name = file.name


def retrieve_context(question: str, k: int = 5) -> tuple[str, list[int]]:
    """
    Return a context string with the top-k most similar chunks and their indices.
    """
    if "pdf_matrix" not in st.session_state:
        return "", []

    vectorizer = st.session_state.pdf_vectorizer
    matrix = st.session_state.pdf_matrix
    chunks = st.session_state.pdf_chunks

    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()
    order = scores.argsort()[::-1][:k]

    excerpts = []
    for rank, idx in enumerate(order, start=1):
        score = scores[idx]
        excerpt = chunks[idx]
        header = f"[{rank}] (score={score:.3f})"
        excerpts.append(f"{header}\n{excerpt}")

    context = f"PDF: {st.session_state.get('pdf_name','(uploaded)')}\n\n" + "\n\n-----\n\n".join(excerpts)
    return context, order.tolist()


# -----------------------------
# Build Strands Agent (LiteLLM → Gemini)
# -----------------------------
litellm_model = LiteLLMModel(
    client_args={"api_key": GEMINI_API_KEY},
    model_id=model_id,  # e.g., "gemini/gemini-2.0-flash"
    params={"temperature": temperature},
)

agent = Agent(
    model=litellm_model,
    system_prompt=sys_prompt,  # keep system-level policy here
)


# -----------------------------
# Chat utilities
# -----------------------------
def to_strands_messages(history: list[dict]) -> list[dict]:
    """
    Convert [{'role': 'user'|'assistant', 'content': str}, ...] to Strands Messages:
      { "role": "user"|"assistant", "content": [ {"text": "..."} ] }
    Drops blanks & coerces content to strings.
    """
    msgs: list[dict] = []
    for m in history:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            continue
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": [{"text": content}]})
    return msgs


def build_rag_prompt(question: str, context: str) -> str:
    """
    Create a single user message combining the PDF context + question.
    The system prompt already says to only use provided excerpts.
    """
    return (
        "You are given PDF excerpts below. Use ONLY these to answer.\n"
        "If the answer isn't clearly contained, reply exactly: not sure.\n\n"
        f"=== PDF EXCERPTS START ===\n{context}\n=== PDF EXCERPTS END ===\n\n"
        f"Question: {question}"
    )


def run_agent(messages: list[dict]) -> str:
    """Non-streaming fallback."""
    res = agent(messages)
    return getattr(res, "text", str(res))


async def stream_agent(messages: list[dict], write_fn) -> str:
    """
    Stream tokens from Strands async iterator.
    Normalizes event payloads to incremental text updates.
    """
    acc = ""
    async for event in agent.stream_async(messages):
        data = None
        if isinstance(event, dict):
            data = event.get("data")
            if not data and isinstance(event.get("delta"), dict):
                data = event["delta"].get("text")
        if data:
            acc += data
            write_fn(acc)
    return acc


# -----------------------------
# Render history
# -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Build / refresh the PDF index if a file was provided
ensure_pdf_index(uploaded_pdf)

# Chat input
user_input = st.chat_input("Ask a question about the uploaded PDF (or chat normally)…")


# -----------------------------
# Handle new user message
# -----------------------------
if user_input:
    # Show user turn
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # If we have a PDF index, do retrieval and inject context;
    # otherwise, just pass through the normal history.
    have_pdf = "pdf_matrix" in st.session_state
    if have_pdf:
        context, hits = retrieve_context(user_input, k=5)
        # Compose payload: prior history (excluding this last user),
        # then a single user message that includes context + question.
        prior = st.session_state.messages[:-1]
        prior_payload = to_strands_messages(prior)
        augmented_user = {
            "role": "user",
            "content": [{"text": build_rag_prompt(user_input, context)}],
        }
        payload = [*prior_payload, augmented_user]
    else:
        payload = to_strands_messages(st.session_state.messages)

    # Stream model response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            answer = asyncio.run(stream_agent(payload, placeholder.markdown))
        except RuntimeError:
            answer = run_agent(payload)
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"⚠️ Error: {e}"
            placeholder.markdown(answer)

    # Append assistant turn
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # (Optional) Tiny provenance footer when using PDF context
    if have_pdf and isinstance(answer, str) and answer.strip():
        with st.expander("🔎 Context used (top matches)"):
            st.markdown(
                "The answer above was generated using the most similar excerpts from the uploaded PDF."
            )
            st.code(context[:5000])  # avoid overly long UI blocks
