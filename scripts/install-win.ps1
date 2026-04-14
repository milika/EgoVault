#Requires -Version 5.1
# EgoVault — one-liner installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex
#
# What this script does:
#   1. Verifies Python 3.11+ is available
#   2. Installs pipx if not present
#   3. Installs (or upgrades) egovault via pipx
#   4. Writes a default egovault.toml to ~\.config\egovault\
#   5. Creates inbox and data directories
#
# Safe to re-run — existing config is never overwritten.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# ── helper output functions ───────────────────────────────────────────────────
function Write-Info  { param($m) Write-Host "[info]  $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "[ok]    $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "[warn]  $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "[error] $m" -ForegroundColor Red; throw $m }

# ── 1. locate Python 3.11+ with pip ──────────────────────────────────────────
$python = $null
foreach ($cmd in @('python3.13', 'python3.12', 'python3.11', 'python3', 'python')) {
    $exe = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($exe) {
        $ver = & $exe.Path -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)" 2>$null
        if (($ver -as [int]) -ge 311) {
            # Skip interpreters without pip (e.g. MSYS2, store stub)
            $hasPip = & $exe.Path -m pip --version 2>$null
            if (-not $hasPip) {
                Write-Warn "Skipping $($exe.Path) — no pip available"
                continue
            }
            $python = $exe.Path
            break
        }
    }
}

if (-not $python) {
    Write-Fail @"
Python 3.11 or later with pip is required but was not found.
  Install from python.org (recommended):
    winget install Python.Python.3.12
  or:  https://www.python.org/downloads/windows/
  Make sure to check 'Add Python to PATH' during installation.
  Note: MSYS2/Cygwin Python is not supported — install the official python.org build.
"@
}

Write-Ok "Python: $(& $python --version)"

# ── 2. ensure pipx ────────────────────────────────────────────────────────────
$pipxCmd = Get-Command pipx -ErrorAction SilentlyContinue

if (-not $pipxCmd) {
    Write-Info "pipx not found — installing..."
    & $python -m pip install --user --quiet pipx

    # Refresh PATH from current user environment variables.
    $userPath = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
    if ($userPath) {
        $env:PATH = "$userPath;$env:PATH"
    }
    # Also add the Python user Scripts directory explicitly.
    $userScripts = & $python -c "import site; print(site.getusersitepackages())" 2>$null
    if ($userScripts) {
        $userScripts = Split-Path $userScripts -Parent | Join-Path -ChildPath 'Scripts'
        $env:PATH = "$userScripts;$env:PATH"
    }

    $pipxCmd = Get-Command pipx -ErrorAction SilentlyContinue
    if (-not $pipxCmd) {
        Write-Fail @"
pipx was installed but is not in PATH.
  Run:  $python -m pipx ensurepath
  Then restart PowerShell and re-run this script.
"@
    }
}

Write-Ok "pipx $(& pipx --version)"

# ── 3. install / upgrade egovault ─────────────────────────────────────────────
$pipxList = & pipx list 2>$null
if ($pipxList -match 'package egovault') {
    Write-Info "egovault already installed — upgrading..."
    & pipx upgrade egovault
} else {
    Write-Info "Installing egovault..."
    & pipx install egovault
}

$evVer = & egovault --version 2>$null
Write-Ok "egovault $evVer"

# ── 4. paths ──────────────────────────────────────────────────────────────────
$configDir  = Join-Path $HOME '.config\egovault'
$configFile = Join-Path $configDir 'egovault.toml'
$dataDir    = Join-Path $HOME '.local\share\egovault'
$inboxDir   = Join-Path ([System.Environment]::GetFolderPath('MyDocuments')) 'egovault-inbox'

# Use TOML-safe forward-slash paths.
$dataDirFwd  = $dataDir  -replace '\\', '/'
$inboxDirFwd = $inboxDir -replace '\\', '/'

New-Item -ItemType Directory -Force -Path $configDir              | Out-Null
New-Item -ItemType Directory -Force -Path "$dataDir\models"       | Out-Null
New-Item -ItemType Directory -Force -Path "$dataDir\output"       | Out-Null
New-Item -ItemType Directory -Force -Path $inboxDir               | Out-Null

# ── 5. write default config (only if missing) ─────────────────────────────────
if (Test-Path $configFile) {
    Write-Warn "Config already exists at $configFile — skipping."
} else {
    @"
[general]
vault_db   = "$dataDirFwd/vault.db"
inbox_dir  = "$inboxDirFwd"
output_dir = "$dataDirFwd/output"

[llm]
provider = "llama_cpp"
model    = "gemma-4-e2b-it"
base_url = "http://127.0.0.1:8080"

[llama_cpp]
manage          = true
model_path      = "$dataDirFwd/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
model_hf_repo   = "unsloth/gemma-4-E2B-it-GGUF"
flash_attn      = true
vram_budget_pct = 0.80

[reranker]
enabled = true
backend = "auto"

[web_search]
provider    = "duckduckgo"
max_results = 5
"@ | Set-Content -Path $configFile -Encoding UTF8

    Write-Ok "Config written to $configFile"
}

# ── done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "EgoVault is ready!" -ForegroundColor White -BackgroundColor DarkGreen
Write-Host "  vault DB  :  $dataDir\vault.db"
Write-Host "  inbox     :  $inboxDir"
Write-Host "  config    :  $configFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Drop files into $inboxDir"
Write-Host "  2. egovault chat       # terminal REPL"
Write-Host "  3. egovault web        # Streamlit browser UI"
Write-Host ""
Write-Host "Full docs: https://github.com/milika/EgoVault/blob/main/docs/installation.md"
Write-Host ""

# Remind the user if egovault is not yet on PATH.
if (-not (Get-Command egovault -ErrorAction SilentlyContinue)) {
    Write-Warn "'egovault' not yet in PATH — run: pipx ensurepath  then restart PowerShell."
}
