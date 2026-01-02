# Install Visual Studio Build Tools (C++ compiler)
# Run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visual Studio Build Tools Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "This will install Visual Studio Build Tools 2022" -ForegroundColor Yellow
Write-Host "Required components:" -ForegroundColor Yellow
Write-Host "  - MSVC v143 C++ x64/x86 build tools" -ForegroundColor White
Write-Host "  - Windows SDK" -ForegroundColor White
Write-Host "  - CMake tools for Windows" -ForegroundColor White
Write-Host ""
Write-Host "Installation size: ~6 GB" -ForegroundColor Gray
Write-Host "Installation time: ~10-20 minutes" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Continue with installation? (Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Installation cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Installing Visual Studio Build Tools 2022..." -ForegroundColor Green
Write-Host ""

# Use Chocolatey to install VS Build Tools
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart" -y

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visual Studio Build Tools Installed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Restart your computer for changes to take effect" -ForegroundColor Yellow
Write-Host ""
Write-Host "After restart, run: .\build_lightgbm_dll.ps1" -ForegroundColor White
Write-Host ""

pause
