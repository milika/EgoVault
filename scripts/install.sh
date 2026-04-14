#!/usr/bin/env sh
# EgoVault - one-liner installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install.sh | sh
#
# What this script does:
#   1. Verifies Python 3.11+ (with pip) is available
#   2. Creates an egovault/ folder in the current directory
#   3. Creates a .venv inside that folder
#   4. Installs egovault into the venv
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

# -- 1. locate Python 3.11+ (pip not required - venv bootstraps its own) ------
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
  macOS:   brew install python                              (https://brew.sh)
  Ubuntu:  sudo apt install python3.13 python3.13-venv
  Other:   https://www.python.org/downloads/"
fi
ok "Python: $($PYTHON --version)"

# -- 2. create install folder in current directory ----------------------------
INSTALL_DIR="$(pwd)/egovault"
VENV_DIR="$INSTALL_DIR/.venv"
VENV_BIN="$VENV_DIR/bin"
VENV_PIP="$VENV_BIN/pip"
EV_EXE="$VENV_BIN/egovault"

mkdir -p "$INSTALL_DIR"
info "Install directory: $INSTALL_DIR"

# -- 3. create / reuse venv ---------------------------------------------------
if [ -f "$VENV_BIN/python" ]; then
    info "Using existing venv at $VENV_DIR"
else
    info "Creating venv ..."
    if ! "$PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        # Detect Python minor version for the package name (e.g. python3.13-venv)
        py_minor=$("$PYTHON" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}-venv')" 2>/dev/null || true)
        if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
            warn "venv not available - attempting: sudo apt-get install -y $py_minor"
            sudo apt-get install -y "$py_minor" >/dev/null 2>&1 || \
                die "Auto-install failed. Run manually: sudo apt-get install -y $py_minor"
            info "Retrying venv creation ..."
            "$PYTHON" -m venv "$VENV_DIR" || \
                die "venv creation failed even after installing $py_minor"
        else
            die "venv creation failed. Install the venv package first:
  Ubuntu/Debian: sudo apt-get install -y ${py_minor:-python3-venv}"
        fi
    fi
fi

# Bootstrap pip inside the venv if it wasn't bundled (common on Ubuntu minimal)
if ! "$VENV_PIP" --version >/dev/null 2>&1; then
    info "Bootstrapping pip inside venv ..."
    if "$VENV_BIN/python" -m ensurepip --upgrade 2>/dev/null; then
        : # ensurepip worked
    else
        info "ensurepip unavailable - fetching get-pip.py (no sudo needed) ..."
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV_BIN/python" || \
                die "pip bootstrap via get-pip.py failed"
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://bootstrap.pypa.io/get-pip.py | "$VENV_BIN/python" || \
                die "pip bootstrap via get-pip.py failed"
        else
            die "Cannot bootstrap pip: curl/wget not found. Install python3.13-venv or pip manually."
        fi
    fi
fi

# -- 4. install / upgrade egovault into the venv ------------------------------
if "$VENV_PIP" show egovault >/dev/null 2>&1; then
    info "egovault already installed - upgrading..."
    "$VENV_PIP" install --upgrade --no-cache-dir --progress-bar on egovault
else
    info "Installing egovault..."
    "$VENV_PIP" install --no-cache-dir --progress-bar on egovault
fi
ok "egovault $($EV_EXE --version 2>/dev/null || true)"

# -- 5. create data dirs and config inside the install folder -----------------
DATA_DIR="$INSTALL_DIR/data"
INBOX_DIR="$INSTALL_DIR/inbox"
CONFIG_FILE="$INSTALL_DIR/egovault.toml"

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
printf '#!/usr/bin/env sh\nexec "%s" "$@"\n' "$EV_EXE" > "$LAUNCHER"
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

# Launch ego chat immediately
exec "$EV_EXE" chat
