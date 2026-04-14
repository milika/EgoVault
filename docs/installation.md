# EgoVault — Installation & Configuration

## Install

### One-liner (recommended)

Each script installs `pipx`, pulls `egovault`, and writes a default `egovault.toml`. Safe to re-run.

**Windows — PowerShell** (Windows 10/11, PowerShell 5+)
```powershell
irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex
```

**Windows — Command Prompt** (calls PowerShell automatically)
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex"
```

**Linux / WSL — bash or sh**
```bash
curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
```

**macOS — zsh (default) or bash**
```zsh
curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
```

### From source (contributors / pre-release)

**Windows — PowerShell**
```powershell
git clone https://github.com/milika/EgoVault; cd egovault; python -m venv .venv; .venv\Scripts\pip install -e ".[local]"
```

**Windows — Command Prompt**
```cmd
git clone https://github.com/milika/EgoVault && cd egovault && python -m venv .venv && .venv\Scripts\pip install -e ".[local]"
```

**Linux / macOS — bash or zsh**
```bash
git clone https://github.com/milika/EgoVault && cd egovault && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[local]"
```

Then launch with the included wrapper:
```bash
# Linux / macOS
./ego.sh chat

# Windows PowerShell
.\ego.ps1 chat

# Windows Command Prompt
ego.cmd chat
```

### pipx (any platform, Python 3.11+ required)

```bash
pipx install egovault
```

### Slim down (optional)

The default install includes all modules. Remove what you don't need:

```bash
pip uninstall streamlit
pip uninstall mcp
pip uninstall sentence-transformers
pip uninstall google-auth-oauthlib google-api-python-client
pip uninstall pdfplumber python-docx beautifulsoup4 lxml ebooklib pandas openpyxl python-pptx
```

---

## Configuration

EgoVault is configured via `egovault.toml` in the project root. A minimal config:

```toml
[general]
vault_db   = "./data/vault.db"
inbox_dir  = "./inbox"

[llm]
provider = "llama_cpp"       # or "openai" for any OpenAI-compatible API
model    = "gemma-4-e2b-it"
base_url = "http://127.0.0.1:8080"

[llama_cpp]
manage          = true                               # auto-start llama-server
model_path      = "./models/gemma-4-e2b-it-Q4_K_XL.gguf"
model_hf_repo   = "unsloth/gemma-4-e2b-it-GGUF"    # auto-download source
flash_attn      = true
vram_budget_pct = 0.80                              # KV cache gets 80 % of free VRAM

[reranker]
enabled = true
backend = "bm25"             # or "cross-encoder" (requires sentence-transformers)

[web_search]
provider    = "duckduckgo"   # "duckduckgo" | "searxng" | "" (disabled)
max_results = 5
```

To use a cloud provider instead:

```toml
[llm]
provider = "openai"
base_url = "https://api.openai.com"
api_key  = "sk-..."
model    = "gpt-4o-mini"
```

---

## llama-server setup

EgoVault uses [llama-server](https://github.com/ggml-org/llama.cpp) as its local LLM backend.

With `[llama_cpp] manage = true` (the default), **EgoVault starts llama-server automatically** and downloads the GGUF model on first run. No manual steps needed.

**To use managed startup:**
1. Install `llama-server` from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and add it to PATH.
2. Run `egovault chat` — GGUF auto-downloads from HuggingFace on first run.

**Manual startup (set `manage = false`):**

```bash
llama-server \
  -m path/to/gemma-4-e2b-it-Q4_K_XL.gguf \
  --n-gpu-layers 99 \
  --ctx-size 65536 \
  --flash-attn \
  --host 127.0.0.1 --port 8080 \
  --embedding --pooling mean
```

EgoVault queries free VRAM via `nvidia-smi` and auto-computes `ctx-size` so the KV cache uses up to `vram_budget_pct` of whatever is free. `--flash-attn` roughly halves KV cache VRAM, doubling effective context.

| GPU VRAM | Model | ctx-size (auto) | Total VRAM used |
|----------|-------|-----------------|----------------|
| 12 GB (RTX 3080 Ti) | e2b Q4_K_XL (~3 GB) | 65536 | ~7 GB (57 %) ✓ |
| 12 GB (RTX 3080 Ti) | e4b Q4_K_XL (~7.5 GB) | 32768 | ~9 GB (75 %) ✓ |
| 16 GB | Q6_K (~9 GB) | 32768 | ~11 GB ✓ |
| 24 GB | Q8_0 (~13 GB) | 65536 | ~17 GB ✓ |

Download Unsloth Dynamic GGUF models from [unsloth.ai](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).

---

## Remote access (WAN tunnel)

`egovault web` auto-starts a [localhost.run](https://localhost.run) SSH tunnel so the Streamlit UI is reachable from any device — phone, laptop, remote desktop.

**First run:** a key pair is generated at `data/wan_id_ed25519`. The terminal prints the public key and the registration URL.

**One-time setup (free):**
1. Create an account at https://admin.localhost.run
2. Add the printed `ssh-ed25519 ...` public key to your account
3. Every `egovault web` run now produces the **same URL forever**

**Set a WAN password** (required before first `--wan` use):
```bash
egovault web-password
```

Use `--no-wan` to disable the tunnel entirely (LAN only).

---

## CLI reference

| Command | Description |
|---------|-------------|
| `ego chat` | Interactive RAG chat in the terminal |
| `ego web [--wan\|--no-wan]` | Streamlit browser UI; WAN tunnel on by default |
| `ego web-password` | Set the WAN access password |
| `ego mcp` | Start the MCP server (connect AnythingLLM as Custom Agent) |
| `ego scan` | Ingest files from the configured inbox |
| `ego scan-folder <path>` | Ingest a specific folder |
| `ego embed` | Generate embeddings for all chunks |
| `ego chunk` | Rechunk stored records |
| `ego context` | Inspect retrieval context for a query |
| `ego gmail-auth` | Authenticate Gmail via OAuth |
| `ego gmail-sync` | Sync Gmail inbox into the vault |
| `ego export` | Export vault records to CSV/Markdown |

---

## Cross-platform notes

### Windows
- Use `ego.ps1` or `ego.cmd` as the launcher
- GPU auto-detected via `nvidia-smi`; CPU fallback if absent
- One-liner: `irm <url>/install-win.ps1 | iex`

### Linux
- `ego.sh` must be executable (`chmod +x ego.sh`)
- GPU auto-detected via `nvidia-smi`
- One-liner: `curl -fsSL <url>/install.sh | sh`
- Systemd: copy `scripts/egovault-web.service` to `~/.config/systemd/user/` for auto-start

### macOS
- GPU acceleration via Metal; no `nvidia-smi` needed
- One-liner: `curl -fsSL <url>/install.sh | sh`
- Login auto-start: copy `scripts/io.github.egovault.web.plist` to `~/Library/LaunchAgents/`
