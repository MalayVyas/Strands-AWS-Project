✨ Features

🧠 Strands Agent orchestration (model-agnostic; easy to add tools later)

⚡ Gemini via LiteLLM (configurable model ID; quick swaps)

🖥️ Streamlit UI with chat history

🧵 Streaming responses with non-streaming fallback

🔒 Local-first: only outbound requests to the model provider

📦 Tech Stack

UI: Streamlit

Agent runtime: Strands

Model provider: LiteLLM → Google Gemini

Config: .env or Streamlit Secrets

📁 Project Structure
.
├─ app.py
├─ requirements.txt
├─ .env                      # not committed; holds API key and defaults
└─ .streamlit/
   └─ secrets.toml           # optional alternative to .env (local only)

🚀 Quickstart

Python requirement: 3.10+ (3.11+ recommended)

1) Clone & enter
git clone https://github.com/<yourname>/strands-gemini-chat.git
cd strands-gemini-chat

2) Create a virtual environment

Windows (PowerShell):

py -3.11 -m venv TestingEnv
.\TestingEnv\Scripts\Activate.ps1


If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\TestingEnv\Scripts\Activate.ps1


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3) Install dependencies
python -m pip install -U pip
pip install -r requirements.txt

4) Add your API key

Create a .env file at the project root:

# .env
GEMINI_API_KEY=your-gemini-key-here
# or: GOOGLE_API_KEY=your-gemini-key-here

# Optional defaults
MODEL_ID=gemini/gemini-2.0-flash
TEMPERATURE=0.3


Alternative (Streamlit Secrets):

Create .streamlit/secrets.toml:

GEMINI_API_KEY = "your-gemini-key-here"


or set via Streamlit UI (⋮ → Settings → Secrets).

5) Run the app
streamlit run app.py


Open the printed local URL (usually http://localhost:8501) and chat away. 🎉

⚙️ Configuration

You can change these in the sidebar at runtime or via .env:

MODEL_ID (LiteLLM format), e.g.:

gemini/gemini-2.0-flash (default)

gemini/gemini-2.5-pro

gemini/gemini-1.5-flash

TEMPERATURE — creativity vs determinism (0.0–1.0)

System prompt — controls agent behavior (editable in UI)

🧩 How It Works
Streamlit UI  →  Strands Agent  →  LiteLLM Model Provider  →  Gemini API
   (history)        (orchestration)         (adapter)             (LLM)


We create a Strands Agent with a system_prompt and a LiteLLM model configured for Gemini.

Messages from the UI are converted into Strands’ schema:

{ "role": "user" | "assistant", "content": [ { "text": "..." } ] }


We stream tokens via agent.stream_async(...), with a safe fallback to agent(...) on platforms where async loops are constrained.

🔧 Troubleshooting

PowerShell: “running scripts is disabled”
Execution policy blocked activation.

Temporary (current shell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Per user:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Python version shows 3.8 inside venv
You created the venv with 3.8. Recreate with 3.11+:

py -3.11 -m venv TestingEnv
.\TestingEnv\Scripts\Activate.ps1
python -V


Streamlit: “No secrets found … secrets.toml”
Accessing st.secrets without a secrets file triggers this. Use .env or create .streamlit/secrets.toml.

ImportError: cannot import name 'Message' from 'strands.types'
Newer Strands organizes types differently. This app does not import Message; it builds message dicts in the required schema.

Error: content_type=<Y> | unsupported type
Occurs if content isn’t wrapped as { "content": [ { "text": "..." } ] }, or if it’s empty/non-string. This app converts history and drops blanks.

🛠️ Extending

Add tools (file reader, web fetch, etc.) via Strands tool packages.

Swap providers by changing MODEL_ID to any LiteLLM-supported model (e.g., openai/gpt-4o-mini, groq/llama-3.1-70b) and setting the matching API key.

Observability: wire Strands telemetry to inspect traces/tokens.

Persistence: save st.session_state.messages to disk and add a “Download chat” button.

🧪 Development Notes

Keep dependencies updated:

pip install -U pip
pip install -U -r requirements.txt


If you use VS Code, select the interpreter at:
Ctrl+Shift+P → Python: Select Interpreter → ./TestingEnv/Scripts/python.exe (Windows) or ./.venv/bin/python (macOS/Linux).

🤝 Contributing

PRs and issues are welcome!

Prefer clear commit messages.

Optional: add ruff/black for formatting/linting.

📝 License

MIT — see LICENSE (add one to the repo).

📷 Screenshots

Add your screenshots to docs/ and link them here.

docs/screenshot-1.png — Home screen
docs/screenshot-2.png — Streaming reply

🙏 Credits

Strands
 — agent runtime

LiteLLM
 — model adapter

Streamlit
 — UI framework