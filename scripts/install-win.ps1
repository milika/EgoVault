#Requires -Version 5.1
# EgoVault - one-liner installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex
#
# No Python installation required - uv downloads its own Python runtime.
#
# What this script does:
#   1. Creates an egovault folder in the current directory
#   2. Downloads uv (a single .exe - no Python needed)
#   3. Uses uv to create a .venv with a self-contained Python 3.12
#   4. Installs egovault into the venv via uv
#   5. Writes egovault.toml, inbox/, data/ all inside the folder
#   6. Creates ego.cmd launcher in the folder
#
# Run this from wherever you want EgoVault to live, e.g.:
#   cd C:\Users\you\Projects
#   irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex
#
# Safe to re-run - existing config and venv are never clobbered.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# -- helper output functions ---------------------------------------------------
function Write-Info  { param($m) Write-Host "[info]  $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "[ok]    $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "[warn]  $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "[error] $m" -ForegroundColor Red; throw $m }

# -- 1. create install folder in current directory ----------------------------
$installDir  = Join-Path $PWD 'egovault'
$venvDir     = Join-Path $installDir '.venv'
$venvScripts = Join-Path $venvDir 'Scripts'
$uvExe       = Join-Path $installDir 'uv.exe'

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Write-Info "Install directory: $installDir"

# -- 2. download uv (no Python required) --------------------------------------
if (Test-Path $uvExe) {
    Write-Info "uv already present."
} else {
    Write-Info "Downloading uv..."
    $arch = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq
                [System.Runtime.InteropServices.Architecture]::Arm64) { 'aarch64' } else { 'x86_64' }
    $uvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip"
    $uvZip = Join-Path $env:TEMP 'uv-win.zip'
    $uvTmp = Join-Path $env:TEMP 'uv-win-extract'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $uvZipUrl -OutFile $uvZip -UseBasicParsing
        if (Test-Path $uvTmp) { Remove-Item $uvTmp -Recurse -Force }
        Expand-Archive -Path $uvZip -DestinationPath $uvTmp -Force
        Copy-Item (Join-Path $uvTmp 'uv.exe') $uvExe -Force
    } finally {
        if (Test-Path $uvZip) { Remove-Item $uvZip -Force -ErrorAction SilentlyContinue }
        if (Test-Path $uvTmp) { Remove-Item $uvTmp -Recurse -Force -ErrorAction SilentlyContinue }
    }
    Write-Ok "uv downloaded."
}

# -- 3. create venv with self-contained Python (auto-downloaded by uv) --------
# UV_PYTHON_INSTALL_DIR keeps the Python runtime inside the install folder.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $installDir '.python'

if (Test-Path (Join-Path $venvScripts 'python.exe')) {
    Write-Info "Using existing venv at $venvDir"
} else {
    Write-Info "Creating venv with Python 3.12 (uv will download Python if needed)..."
    & $uvExe venv $venvDir --python cpython-3.12
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create venv." }
}
$venvPython = Join-Path $venvScripts 'python.exe'
Write-Ok "Python: $(& $venvPython --version)"

# -- 4. install / upgrade egovault -------------------------------------------
$evExe = Join-Path $venvScripts 'egovault.exe'
if (Test-Path $evExe) {
    Write-Info "egovault already installed - upgrading..."
    & $uvExe pip install --python $venvPython --upgrade egovault
} else {
    Write-Info "Installing egovault..."
    & $uvExe pip install --python $venvPython egovault
}
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install egovault failed." }

$evVer = & $evExe --version 2>$null
Write-Ok "egovault $evVer"

# -- 5. create data dirs and config inside the install folder -----------------
$dataDir   = Join-Path $installDir 'data'
$inboxDir  = Join-Path $installDir 'inbox'
$configFile = Join-Path $dataDir 'egovault.toml'

$dataDirFwd  = $dataDir  -replace '\\', '/'
$inboxDirFwd = $inboxDir -replace '\\', '/'

New-Item -ItemType Directory -Force -Path $dataDir            | Out-Null
New-Item -ItemType Directory -Force -Path "$dataDir\models"   | Out-Null
New-Item -ItemType Directory -Force -Path "$dataDir\output"   | Out-Null
New-Item -ItemType Directory -Force -Path $inboxDir           | Out-Null

# -- 6. write default config (only if missing) --------------------------------
if (Test-Path $configFile) {
    Write-Warn "Config already exists at $configFile - skipping."
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
"@ | Set-Content -Path $configFile -Encoding ASCII

    Write-Ok "Config written to $configFile"
}

# -- 7. create ego.cmd launcher in current directory -------------------------
$launcherPath = Join-Path $PWD 'ego.cmd'
$launcherContent = "@echo off`r`ncd /d `"$installDir`"`r`n`"$evExe`" %*`r`n"
[System.IO.File]::WriteAllText($launcherPath, $launcherContent, [System.Text.Encoding]::ASCII)
Write-Ok "Launcher created: $launcherPath"

# -- done ----------------------------------------------------------------------
Write-Host ""
Write-Host "EgoVault is ready!" -ForegroundColor White -BackgroundColor DarkGreen
Write-Host "  folder :  $installDir"
Write-Host "  inbox  :  $inboxDir"
Write-Host "  config :  $configFile"
Write-Host ""
Write-Host "To start (from this folder):"
Write-Host "  ego chat       # terminal REPL"
Write-Host "  ego web        # Streamlit browser UI"
Write-Host ""
Write-Host "Full docs: https://github.com/milika/EgoVault/blob/main/docs/installation.md"

# Launch ego chat immediately (cd so egovault.toml is found)
Set-Location $installDir
& $evExe chat
