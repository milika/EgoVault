#!/usr/bin/env sh
# EgoVault — one-liner installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
#
# What this script does:
#   1. Verifies Python 3.11+ is available
#   2. Installs pipx if not present
#   3. Installs (or upgrades) egovault via pipx
#   4. Writes a default egovault.toml to ~/.config/egovault/
#   5. Creates inbox and data directories
#
# Safe to re-run — existing config is never overwritten.

set -e

# ── colours (disabled when not a tty) ─────────────────────────────────────────
if [ -t 1 ]; then
    BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
    YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
else
    BOLD=''; GREEN=''; CYAN=''; YELLOW=''; RED=''; RESET=''
fi
info() { printf "${CYAN}[info]${RESET}  %s\n" "$*"; }
ok()   { printf "${GREEN}[ok]${RESET}    %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
die()  { printf "${RED}[error]${RESET} %s\n" "$*" >&2; exit 1; }

# ── 1. locate Python 3.11+ ────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        py_ver=$("$cmd" -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)" 2>/dev/null || true)
        if [ -n "$py_ver" ] && [ "$py_ver" -ge 311 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    die "Python 3.11 or later is required.
  macOS:   brew install python           (https://brew.sh)
  Ubuntu:  sudo apt install python3.11
  Other:   https://www.python.org/downloads/"
fi
ok "Python: $($PYTHON --version)"

# ── 2. ensure pipx ────────────────────────────────────────────────────────────
if ! command -v pipx >/dev/null 2>&1; then
    info "pipx not found — installing..."
    "$PYTHON" -m pip install --user --quiet pipx

    # Add the user script directory to PATH for the rest of this session.
    USER_BIN="$($PYTHON -m site --user-base)/bin"
    export PATH="$PATH:$USER_BIN"

    if ! command -v pipx >/dev/null 2>&1; then
        "$PYTHON" -m pipx ensurepath >/dev/null 2>&1 || true
        die "pipx was installed but is not in PATH.
  Add '$USER_BIN' to your PATH (or run: $PYTHON -m pipx ensurepath)
  then re-run this script."
    fi
fi
ok "pipx $(pipx --version)"

# ── 3. install / upgrade egovault ─────────────────────────────────────────────
if pipx list 2>/dev/null | grep -qF 'package egovault'; then
    info "egovault already installed — upgrading..."
    pipx upgrade egovault
else
    info "Installing egovault..."
    pipx install egovault
fi
ok "egovault $(egovault --version 2>/dev/null || true)"

# ── 4. paths ──────────────────────────────────────────────────────────────────
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/egovault"
CONFIG_FILE="$CONFIG_DIR/egovault.toml"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/egovault"
INBOX_DIR="$HOME/Documents/egovault-inbox"

mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/output"
mkdir -p "$INBOX_DIR"

# ── 5. write default config (only if missing) ─────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
    warn "Config already exists at $CONFIG_FILE — skipping."
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

# ── done ──────────────────────────────────────────────────────────────────────
printf "\n${BOLD}EgoVault is ready!${RESET}\n"
printf "  vault DB  :  %s\n" "$DATA_DIR/vault.db"
printf "  inbox     :  %s\n" "$INBOX_DIR"
printf "  config    :  %s\n" "$CONFIG_FILE"
printf "\nNext steps:\n"
printf "  1. Drop files into ~/Documents/egovault-inbox\n"
printf "  2. egovault chat       # terminal REPL\n"
printf "  3. egovault web        # Streamlit browser UI\n"
printf "\nFull docs: https://github.com/milika/EgoVault/blob/main/docs/installation.md\n\n"

# Remind the user to reload PATH if pipx was just installed.
if ! command -v egovault >/dev/null 2>&1; then
    warn "'egovault' not yet in PATH — run: source ~/.profile  (or restart your shell)"
fi
