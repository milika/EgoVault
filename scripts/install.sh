#!/usr/bin/env sh
# EgoVault - one-liner installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
#
# No Python installation required - uv downloads its own Python runtime.
#
# What this script does:
#   1. Creates an egovault/ folder in the current directory
#   2. Downloads uv (a single binary - no Python needed)
#   3. Uses uv to create a .venv with a self-contained Python 3.12
#   4. Installs egovault into the venv via uv
#   5. Writes egovault.toml, inbox/, data/ all inside the folder
#   6. Creates an ego launcher script in the current directory
#   7. Launches ego chat
#
# Run this from wherever you want EgoVault to live, e.g.:
#   cd ~/Projects
#   curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
#
# Safe to re-run - existing config and venv are never clobbered.

set -e

# -- colours (disabled when not a tty) ----------------------------------------
if [ -t 1 ]; then
    BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
    YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
else
    BOLD=''; GREEN=''; CYAN=''; YELLOW=''; RED=''; RESET=''
fi
info() { printf "${CYAN}[info]  ${RESET}%s\n" "$*"; }
ok()   { printf "${GREEN}[ok]    ${RESET}%s\n" "$*"; }
warn() { printf "${YELLOW}[warn]  ${RESET}%s\n" "$*"; }
die()  { printf "${RED}[error] ${RESET}%s\n" "$*" >&2; exit 1; }

# -- 1. create install folder -------------------------------------------------
INSTALL_DIR="$(pwd)/egovault"
VENV_DIR="$INSTALL_DIR/.venv"
VENV_BIN="$VENV_DIR/bin"
UV="$INSTALL_DIR/uv"
EV_EXE="$VENV_BIN/egovault"

mkdir -p "$INSTALL_DIR"
info "Install directory: $INSTALL_DIR"

# -- 2. download uv (no Python required) -------------------------------------
if [ -f "$UV" ]; then
    info "uv already present."
else
    info "Downloading uv..."
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    if [ "$OS" = "Darwin" ]; then
        case "$ARCH" in
            arm64)  UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" ;;
            *)      UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz" ;;
        esac
    else
        # Linux - musl build works on any distro without libc version constraints
        case "$ARCH" in
            aarch64|arm64) UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-musl.tar.gz" ;;
            *)             UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz" ;;
        esac
    fi
    UV_TMP="$(mktemp -d)"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$UV_URL" | tar -xz -C "$UV_TMP"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$UV_URL" | tar -xz -C "$UV_TMP"
    else
        die "curl or wget is required to download uv. Install one and retry."
    fi
    mv "$UV_TMP/uv" "$UV"
    rm -rf "$UV_TMP"
    chmod +x "$UV"
    ok "uv downloaded."
fi

# -- 3. create venv with self-contained Python (auto-downloaded by uv) --------
# UV_PYTHON_INSTALL_DIR keeps the Python runtime inside the install folder.
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/.python"

if [ -f "$VENV_BIN/python" ]; then
    info "Using existing venv at $VENV_DIR"
else
    info "Creating venv with Python 3.12 (uv will download Python if needed)..."
    "$UV" venv "$VENV_DIR" --python cpython-3.12
fi
ok "Python: $($VENV_BIN/python --version)"

# -- 4. install / upgrade egovault -------------------------------------------
if [ -f "$EV_EXE" ]; then
    info "egovault already installed - upgrading..."
    "$UV" pip install --python "$VENV_BIN/python" --upgrade egovault
else
    info "Installing egovault..."
    "$UV" pip install --python "$VENV_BIN/python" egovault
fi
ok "egovault $($EV_EXE --version 2>/dev/null || true)"

# -- 5. create data dirs and config inside the install folder -----------------
DATA_DIR="$INSTALL_DIR/data"
INBOX_DIR="$INSTALL_DIR/inbox"
CONFIG_FILE="$DATA_DIR/egovault.toml"

mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/output"
mkdir -p "$INBOX_DIR"

# -- 6. write default config (only if missing) --------------------------------
if [ -f "$CONFIG_FILE" ]; then
    warn "Config already exists at $CONFIG_FILE - skipping."
else
    cat > "$CONFIG_FILE" <<EOF
[general]
vault_db   = "$DATA_DIR/vault.db"
inbox_dir  = "$INBOX_DIR"
output_dir = "$DATA_DIR/output"

[llm]
provider = "llama_cpp"
model    = "gemma-4-e2b-it"
base_url = "http://127.0.0.1:8080"

[llama_cpp]
manage          = true
model_path      = "$DATA_DIR/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
model_hf_repo   = "unsloth/gemma-4-E2B-it-GGUF"
flash_attn      = true
vram_budget_pct = 0.80

[reranker]
enabled = true
backend = "auto"

[web_search]
provider    = "duckduckgo"
max_results = 5
EOF
    ok "Config written to $CONFIG_FILE"
fi

# -- 7. create ego launcher in current directory ------------------------------
LAUNCHER="$(pwd)/ego"
printf '#!/usr/bin/env sh\ncd "%s"\nexec "%s" "$@"\n' "$INSTALL_DIR" "$EV_EXE" > "$LAUNCHER"
chmod +x "$LAUNCHER"
ok "Launcher created: $LAUNCHER"

# -- done ---------------------------------------------------------------------
printf "\n${BOLD}EgoVault is ready!${RESET}\n"
printf "  folder :  %s\n" "$INSTALL_DIR"
printf "  inbox  :  %s\n" "$INBOX_DIR"
printf "  config :  %s\n" "$CONFIG_FILE"
printf "\nTo start (from this folder):\n"
printf "  ./ego chat       # terminal REPL\n"
printf "  ./ego web        # Streamlit browser UI\n"
printf "\nFull docs: https://github.com/milika/EgoVault/blob/main/docs/installation.md\n\n"

# Launch ego chat immediately (cd so egovault.toml is found)
cd "$INSTALL_DIR"
# When piped via `curl | sh`, stdin is the pipe not the terminal.
# Re-attach stdin to /dev/tty so the chat REPL can read user input.
if [ ! -t 0 ] && [ -c /dev/tty ]; then
    exec "$EV_EXE" chat < /dev/tty
else
    exec "$EV_EXE" chat
fi
