import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Strands
from strands import Agent
from strands.models.litellm import LiteLLMModel

st.set_page_config(page_title="Strands × Gemini Chat", page_icon="💬", layout="centered")
load_dotenv()

def resolve_gemini_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if not key:
            key = st.secrets.get("GOOGLE_API_KEY", None)
    except FileNotFoundError:
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

GEMINI_API_KEY = resolve_gemini_key()

# Sidebar
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
        value="You are a concise, helpful assistant. If unsure, reply exactly: not sure.",
        height=120,
    )
    if st.button("🧹 Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

st.title("💬 Strands × Gemini (local)")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Type your message…")

# ---- Helpers ----

def to_strands_messages(history: list[dict]) -> list[dict]:
    """
    Convert chat history to Strands Messages:
      { "role": "user"|"assistant", "content": [ {"text": "..."} ] }
    - Drops blanks/unknown roles
    - Coerces non-strings to strings to avoid content_type issues
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

# Strands Agent (LiteLLM → Gemini)
litellm_model = LiteLLMModel(
    client_args={"api_key": GEMINI_API_KEY},
    model_id=model_id,  # e.g., "gemini/gemini-2.0-flash"
    params={"temperature": temperature},
)

agent = Agent(
    model=litellm_model,
    system_prompt=sys_prompt,  # keep system prompt here; not in the messages list
)

def run_agent(messages: list[dict]) -> str:
    """Non-streaming fallback."""
    res = agent(messages)
    # most providers return an object with .text; keep a safe fallback
    return getattr(res, "text", str(res))

async def stream_agent(messages: list[dict], write_fn) -> str:
    """
    Stream tokens via Strands async iterator.
    Normalizes event payloads to incremental text updates.
    """
    acc = ""
    async for event in agent.stream_async(messages):
        data = None
        if isinstance(event, dict):
            # some providers emit 'data': '...', others 'delta': {'text': '...'}
            data = event.get("data")
            if not data and isinstance(event.get("delta"), dict):
                data = event["delta"].get("text")
        if data:
            acc += data
            write_fn(acc)
    return acc

# Handle new user message
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = to_strands_messages(st.session_state.messages)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            acc = asyncio.run(stream_agent(payload, placeholder.markdown))
        except RuntimeError:
            # event loop already running (rare in Streamlit) → fall back
            acc = run_agent(payload)
            placeholder.markdown(acc)
        except Exception as e:
            acc = f"⚠️ Error: {e}"
            placeholder.markdown(acc)

    st.session_state.messages.append({"role": "assistant", "content": acc})
