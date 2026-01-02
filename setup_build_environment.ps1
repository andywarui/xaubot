# Automated Build Environment Setup for LightGBM MT5 DLL
# Run this script as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LightGBM MT5 DLL - Build Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "[1/4] Checking for Chocolatey package manager..." -ForegroundColor Yellow

# Install Chocolatey if not present
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing Chocolatey..." -ForegroundColor Green
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    Write-Host "  ✓ Chocolatey installed" -ForegroundColor Green
} else {
    Write-Host "  ✓ Chocolatey already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/4] Installing CMake..." -ForegroundColor Yellow

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    Write-Host "  ✓ CMake installed" -ForegroundColor Green
} else {
    Write-Host "  ✓ CMake already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/4] Installing Git (required for vcpkg)..." -ForegroundColor Yellow

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    choco install git -y

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    Write-Host "  ✓ Git installed" -ForegroundColor Green
} else {
    Write-Host "  ✓ Git already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/4] Setting up vcpkg..." -ForegroundColor Yellow

$vcpkgPath = "C:\vcpkg"

if (-not (Test-Path $vcpkgPath)) {
    Write-Host "  Cloning vcpkg repository..." -ForegroundColor Green
    git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath

    Write-Host "  Bootstrapping vcpkg..." -ForegroundColor Green
    & "$vcpkgPath\bootstrap-vcpkg.bat"

    # Add to PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($currentPath -notlike "*$vcpkgPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$vcpkgPath", "Machine")
    }

    Write-Host "  ✓ vcpkg installed at $vcpkgPath" -ForegroundColor Green
} else {
    Write-Host "  ✓ vcpkg already exists at $vcpkgPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Tools Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close and reopen PowerShell to refresh environment" -ForegroundColor White
Write-Host "2. Run: .\install_visual_studio.ps1" -ForegroundColor White
Write-Host "   (or install Visual Studio manually)" -ForegroundColor Gray
Write-Host "3. Run: .\build_lightgbm_dll.ps1" -ForegroundColor White
Write-Host ""

pause
