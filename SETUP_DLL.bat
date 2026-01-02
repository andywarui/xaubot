@echo off
REM Master setup script for LightGBM MT5 DLL
REM Run this as Administrator

echo.
echo ============================================
echo   LightGBM MT5 DLL - Complete Setup
echo ============================================
echo.
echo This will install all required tools and build the DLL.
echo.
echo Steps:
echo   1. Install build tools (CMake, Git, vcpkg)
echo   2. Install Visual Studio Build Tools
echo   3. Build LightGBM DLL
echo   4. Deploy to MT5
echo.
echo IMPORTANT: Run this script as Administrator!
echo Right-click and select "Run as Administrator"
echo.
pause

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click the file and select "Run as Administrator"
    pause
    exit /b 1
)

echo.
echo [STEP 1/3] Installing build tools...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0setup_build_environment.ps1"

if %errorLevel% neq 0 (
    echo ERROR: Build tools installation failed!
    pause
    exit /b 1
)

echo.
echo [STEP 2/3] Installing Visual Studio Build Tools...
echo.
echo IMPORTANT: This will download ~6 GB and take 10-20 minutes
echo.
choice /C YN /M "Continue with Visual Studio installation"
if errorlevel 2 goto skip_vs
if errorlevel 1 goto install_vs

:install_vs
powershell -ExecutionPolicy Bypass -File "%~dp0install_visual_studio.ps1"

if %errorLevel% neq 0 (
    echo ERROR: Visual Studio installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Visual Studio Installed!
echo ============================================
echo.
echo NEXT: Restart your computer, then run:
echo   build_lightgbm_dll.bat
echo.
pause
exit /b 0

:skip_vs
echo.
echo Visual Studio installation skipped.
echo Make sure Visual Studio Build Tools are already installed!
echo.
echo Continue to build DLL? (requires Visual Studio)
choice /C YN /M "Continue"
if errorlevel 2 exit /b 0

echo.
echo [STEP 3/3] Building DLL...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0build_lightgbm_dll.ps1"

if %errorLevel% neq 0 (
    echo ERROR: DLL build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SETUP COMPLETE!
echo ============================================
echo.
echo DLL built and deployed to MT5!
echo.
echo Next: Open MT5 and compile XAUUSD_NeuralBot_DLL.mq5
echo.
pause
