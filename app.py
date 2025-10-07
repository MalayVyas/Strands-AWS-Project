import os
import io
import re
import json
import math
import asyncio
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# Strands (generation via Gemini through LiteLLM)
from strands import Agent
from strands.models.litellm import LiteLLMModel

# PDF + retrieval
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gemini Embeddings (optional)
try:
    import google.generativeai as genai
    HAS_GEM_EMB = True
except Exception:
    HAS_GEM_EMB = False


# =========================
# App setup
# =========================
st.set_page_config(page_title="Strands × Gemini — Multi-PDF QA", page_icon="📚", layout="wide")
load_dotenv()

def resolve_gemini_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None) or st.secrets.get("GOOGLE_API_KEY", None)
    except FileNotFoundError:
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

GEMINI_API_KEY = resolve_gemini_key()
if HAS_GEM_EMB and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass  # handle at call time


# =========================
# Sidebar: settings & data
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add it to .env or .streamlit/secrets.toml.")

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
        value=(
            "You answer using ONLY the provided PDF excerpts.\n"
            "If the answer is not clearly present, reply exactly: not sure."
        ),
        height=130,
    )

    st.markdown("---")
    st.subheader("📄 Upload PDFs")
    uploaded_pdfs = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    st.caption("Tip: First index may take a few seconds per PDF.")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🧹 Clear chat"):
            st.session_state.pop("messages", None)
            st.toast("Chat cleared.")
    with colB:
        if st.button("♻️ Clear all PDF indexes"):
            for k in ("docs", "active_docs"):
                st.session_state.pop(k, None)
            st.toast("PDF indexes cleared.")
    with colC:
        st.write("")  # spacer

    st.markdown("---")
    st.subheader("💾 Session")
    # Download current chat
    if "messages" in st.session_state and st.session_state.get("messages"):
        export_json = json.dumps({"messages": st.session_state["messages"]}, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Download chat (JSON)",
            data=export_json.encode("utf-8"),
            file_name="chat_session.json",
            mime="application/json",
        )

    # Load chat
    load_file = st.file_uploader("Load chat (JSON)", type=["json"], accept_multiple_files=False, key="load_chat")
    if load_file:
        try:
            loaded = json.loads(load_file.read().decode("utf-8"))
            if isinstance(loaded, dict) and "messages" in loaded and isinstance(loaded["messages"], list):
                st.session_state["messages"] = loaded["messages"]
                st.toast("Chat loaded.")
            else:
                st.warning("Invalid chat JSON (expected {'messages': [...]}).")
        except Exception as e:
            st.error(f"Failed to load chat: {e}")


st.title("📚 Strands × Gemini — Ask your PDFs")

# =========================
# State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# docs: { doc_id: {name, chunks, pages, method, vectorizer?, matrix, emb? } }
if "docs" not in st.session_state:
    st.session_state.docs: Dict[str, Dict[str, Any]] = {}

# active_docs: set of doc_ids to include in retrieval
if "active_docs" not in st.session_state:
    st.session_state.active_docs = set()

# =========================
# PDF processing
# =========================
def extract_pdf_pages(file_bytes: bytes) -> List[str]:
    """Return a list of page texts (index = page_number)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages

def chunk_pages(pages: List[str], chunk_chars: int = 1200, overlap: int = 200) -> Tuple[List[str], List[int]]:
    """
    Chunk page texts with overlap; return (chunks, page_numbers).
    Keeps page provenance for display.
    """
    chunks, owners = [], []
    for pno, page_text in enumerate(pages, start=1):
        t = (page_text or "").replace("\r", "")
        n, i = len(t), 0
        while i < n:
            chunk = t[i : i + chunk_chars].strip()
            if len(chunk) > 20:
                chunks.append(chunk)
                owners.append(pno)
            i += chunk_chars - overlap
            if i < 0 or i >= n:
                break
    return chunks, owners

# =========================
# Embeddings & TF-IDF
# =========================
def embed_texts_gemini(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings using Gemini text-embedding-004.
    Returns list of embedding vectors.
    """
    if not (HAS_GEM_EMB and GEMINI_API_KEY):
        raise RuntimeError("Gemini embeddings not available")

    model_name = "models/text-embedding-004"  # gemini API naming
    vecs: List[List[float]] = []
    for t in texts:
        t = t or ""
        try:
            out = genai.embed_content(model=model_name, content=t)
            vecs.append(out["embedding"])
        except Exception as e:
            # Fail fast; caller will fallback to TF-IDF
            raise e
    return vecs

def ensure_doc_index(doc_id: str, name: str, file_bytes: bytes):
    """
    Build (or rebuild) the index for a single PDF:
      - preferred: Gemini embeddings
      - fallback: TF-IDF
    Saves into st.session_state.docs[doc_id]
    """
    pages = extract_pdf_pages(file_bytes)
    chunks, owners = chunk_pages(pages)

    if not chunks:
        st.warning(f"No text extracted from: {name}")
        return

    # Try embeddings first
    use_embeddings = False
    emb_matrix = None
    vectorizer = None
    tfidf_matrix = None
    method = "tfidf"

    try:
        vecs = embed_texts_gemini(chunks)
        # Convert to a simple (n, d) list-of-lists; we'll cosine against it with sklearn
        # You can also use numpy arrays if preferred
        import numpy as np
        emb_matrix = np.array(vecs, dtype="float32")
        use_embeddings = True
        method = "emb"
    except Exception:
        # fall back to TF-IDF
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            max_features=50_000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)

    st.session_state.docs[doc_id] = {
        "name": name,
        "chunks": chunks,
        "pages": owners,
        "method": method,
        "emb_matrix": emb_matrix,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }
    st.session_state.active_docs.add(doc_id)

def maybe_index_new_uploads(files: List[Any]):
    """Index any newly uploaded PDFs."""
    if not files:
        return
    for f in files:
        doc_id = f"{f.name}|{f.size}"
        if doc_id not in st.session_state.docs:
            ensure_doc_index(doc_id, f.name, f.read())

maybe_index_new_uploads(uploaded_pdfs)

# =========================
# Per-document toggles
# =========================
if st.session_state.docs:
    st.subheader("Included documents")
    doc_labels = []
    for did, meta in st.session_state.docs.items():
        label = f"{meta['name']}  ·  {len(meta['chunks'])} chunks  ·  method: {meta['method']}"
        doc_labels.append((did, label))

    default_selection = [did for did, _ in doc_labels if did in st.session_state.active_docs]
    selection = st.multiselect(
        "Choose which documents to include in retrieval:",
        options=[did for did, _ in doc_labels],
        format_func=lambda did: dict(doc_labels)[did],
        default=default_selection,
    )
    st.session_state.active_docs = set(selection)

# =========================
# Retrieval
# =========================
def cosine_sim_dense(query_vec, doc_matrix):
    # query_vec: (d,), doc_matrix: (n, d)
    import numpy as np
    a = doc_matrix @ query_vec  # (n,)
    denom = (np.linalg.norm(doc_matrix, axis=1) * (np.linalg.norm(query_vec) + 1e-12)) + 1e-12
    return a / denom

def highlight_snippet(text: str, query: str, window: int = 420) -> str:
    """
    Return a shortened snippet with simple query-term highlighting.
    Bold words >=3 chars that appear in the query.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(w) >= 3]
    if not words:
        snippet = text[:window]
    else:
        # Find first occurrence of any word, center window around it
        loc = min((text.lower().find(w) for w in words if text.lower().find(w) != -1), default=-1)
        if loc == -1:
            snippet = text[:window]
        else:
            start = max(0, loc - window // 2)
            snippet = text[start : start + window]

    # Bold matches
    def repl(m):
        return f"**{m.group(0)}**"
    for w in sorted(set(words), key=len, reverse=True):
        snippet = re.sub(fr"(?i)\b({re.escape(w)})\b", repl, snippet)
    return snippet.strip()

def retrieve(question: str, k_total: int = 6) -> Tuple[str, List[Tuple[str, int, float]]]:
    """
    Search across all active docs, return a context string and provenance:
    provenance items: (doc_name, page_no, score)
    """
    if not st.session_state.active_docs:
        return "", []

    hits: List[Tuple[str, int, float, str]] = []  # (doc_id, chunk_idx, score, snippet)
    for did in st.session_state.active_docs:
        meta = st.session_state.docs[did]
        chunks, pages, method = meta["chunks"], meta["pages"], meta["method"]

        if method == "emb" and meta["emb_matrix"] is not None:
            # Embed question
            try:
                qv = embed_texts_gemini([question])[0]  # list[float]
                import numpy as np
                qv = np.array(qv, dtype="float32")
                scores = cosine_sim_dense(qv, meta["emb_matrix"])
            except Exception:
                # embeddings down? fallback to TF-IDF on the fly
                vect = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english", max_features=50_000, ngram_range=(1,2))
                mat = vect.fit_transform(chunks)
                scores = cosine_similarity(vect.transform([question]), mat).ravel()
        else:
            vect = meta["vectorizer"]
            mat = meta["tfidf_matrix"]
            if vect is None or mat is None:
                # Shouldn't happen, but guard
                vect = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english", max_features=50_000, ngram_range=(1,2))
                mat = vect.fit_transform(chunks)
            scores = cosine_similarity(vect.transform([question]), mat).ravel()

        # Take top 4 per-doc, we’ll merge globally later
        top_idx = scores.argsort()[::-1][:4]
        for idx in top_idx:
            snippet = highlight_snippet(chunks[idx], question)
            hits.append((did, idx, float(scores[idx]), snippet))

    # Global top-k
    hits.sort(key=lambda x: x[2], reverse=True)
    hits = hits[:k_total]

    # Build context block with provenance
    blocks = []
    provenance: List[Tuple[str, int, float]] = []
    for rank, (did, idx, score, snippet) in enumerate(hits, start=1):
        meta = st.session_state.docs[did]
        name = meta["name"]
        page_no = meta["pages"][idx]
        provenance.append((name, page_no, score))
        blocks.append(
            f"[{rank}] {name} — p.{page_no} (score={score:.3f})\n{snippet}"
        )

    context = "=== RETRIEVED EXCERPTS ===\n" + "\n\n-----\n\n".join(blocks) if blocks else ""
    return context, provenance

def to_strands_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert to Strands schema: {role, content:[{text}]} and drop blanks."""
    msgs: List[Dict[str, Any]] = []
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

def build_rag_user_message(question: str, context: str) -> Dict[str, Any]:
    """Single user turn containing context + question."""
    text = (
        "Use ONLY the excerpts below to answer.\n"
        "If the answer isn't clearly present, reply exactly: not sure.\n\n"
        f"{context}\n\nQuestion: {question}"
    )
    return {"role": "user", "content": [{"text": text}]}

# =========================
# Strands agent
# =========================
litellm_model = LiteLLMModel(
    client_args={"api_key": GEMINI_API_KEY},
    model_id=model_id,
    params={"temperature": temperature},
)

agent = Agent(model=litellm_model, system_prompt=sys_prompt)

def run_agent(messages: List[Dict[str, Any]]) -> str:
    res = agent(messages)
    return getattr(res, "text", str(res))

async def stream_agent(messages: List[Dict[str, Any]], write_fn) -> str:
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

# =========================
# Chat UI
# =========================
# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input row
user_input = st.chat_input("Ask a question about your selected PDFs (or chat normally)…")

# Handle user message
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    have_docs = bool(st.session_state.active_docs)
    if have_docs:
        context, prov = retrieve(user_input, k_total=6)
        prior = st.session_state.messages[:-1]
        prior_payload = to_strands_messages(prior)
        augmented = build_rag_user_message(user_input, context if context else "(no excerpts)")
        payload = [*prior_payload, augmented]
    else:
        payload = to_strands_messages(st.session_state.messages)

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

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Context/provenance panel
    if have_docs and isinstance(answer, str) and answer.strip():
        with st.expander("🔎 Context used (top excerpts with page numbers)"):
            st.markdown(
                "The answer above was generated **only** from the following excerpts."
            )
            st.code(context[:8000])
