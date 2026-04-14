#Requires -Version 5.1
# EgoVault - one-liner installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/milika/EgoVault/main/scripts/install-win.ps1 | iex
#
# What this script does:
#   1. Verifies Python 3.11+ (with pip) is available
#   2. Creates an egovault folder in the current directory
#   3. Creates a .venv inside that folder
#   4. Installs egovault into the venv
#   4. Writes egovault.toml, inbox/, data/ all inside the folder
#   5. Creates ego.cmd launcher in the folder
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

# -- 1. locate Python 3.11+ with pip ------------------------------------------
$python = $null
foreach ($cmd in @('python3.13', 'python3.12', 'python3.11', 'python3', 'python')) {
    $exe = Get-Command $cmd -ErrorAction SilentlyContinue
    if (-not $exe) { continue }

    # Get version number; skip store stubs and interpreters that fail
    $ver = $null
    try {
        $ErrorActionPreference = 'Continue'
        $ver = & $exe.Path -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)" 2>$null
    } catch { $ver = $null }
    finally { $ErrorActionPreference = 'Stop' }
    if (($ver -as [int]) -lt 311) { continue }

    # Skip interpreters without pip (e.g. MSYS2, store stub)
    $hasPip = $false
    try {
        $ErrorActionPreference = 'Continue'
        $null = & $exe.Path -m pip --version 2>&1
        $hasPip = ($LASTEXITCODE -eq 0)
    } catch { $hasPip = $false }
    finally { $ErrorActionPreference = 'Stop' }
    if (-not $hasPip) {
        Write-Warn "Skipping $($exe.Path) - no pip available"
        continue
    }
    $python = $exe.Path
    break
}

if (-not $python) {
    Write-Fail @"
Python 3.11 or later with pip is required but was not found.
  Install from python.org (recommended):
    winget install Python.Python.3.12
  or:  https://www.python.org/downloads/windows/
  Make sure to check 'Add Python to PATH' during installation.
  Note: MSYS2/Cygwin Python is not supported - install the official python.org build.
"@
}

Write-Ok "Python: $(& $python --version)"

# -- 2. create install folder in current directory ----------------------------
$installDir  = Join-Path $PWD 'egovault'
$venvDir     = Join-Path $installDir '.venv'
$venvScripts = Join-Path $venvDir 'Scripts'
$venvPip     = Join-Path $venvScripts 'pip.exe'

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Write-Info "Install directory: $installDir"

# -- 3. create / reuse venv ---------------------------------------------------
if (Test-Path (Join-Path $venvDir 'Scripts\python.exe')) {
    Write-Info "Using existing venv at $venvDir"
} else {
    Write-Info "Creating venv ..."
    & $python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create venv at $venvDir" }
}

# -- 4. install / upgrade egovault into the venv ------------------------------
$installed = $false
try {
    $ErrorActionPreference = 'Continue'
    $null = & $venvPip show egovault 2>&1
    $installed = ($LASTEXITCODE -eq 0)
} catch { $installed = $false }
finally { $ErrorActionPreference = 'Stop' }
if ($installed) {
    Write-Info "egovault already installed - upgrading..."
    & $venvPip install --upgrade --quiet egovault
} else {
    Write-Info "Installing egovault..."
    & $venvPip install --quiet egovault
}
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install egovault failed." }

$evExe = Join-Path $venvScripts 'egovault.exe'
$evVer = & $evExe --version 2>$null
Write-Ok "egovault $evVer"

# -- 5. create data dirs and config inside the install folder -----------------
$dataDir   = Join-Path $installDir 'data'
$inboxDir  = Join-Path $installDir 'inbox'
$configFile = Join-Path $installDir 'egovault.toml'

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
$launcherContent = "@echo off`r`n`"$evExe`" %*`r`n"
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

# Launch ego chat immediately
& $evExe chat
