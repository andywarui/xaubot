@echo off
REM ====================================================================
REM  XAUBOT MT5 Deployment Script
REM  Automatically copies EA and Model files to MT5
REM  Created: 2024-12-28
REM ====================================================================

echo.
echo ====================================================================
echo  XAUBOT Neural Bot v2.0 - MT5 Deployment
echo  Copying files to MT5 installation...
echo ====================================================================
echo.

REM Define paths
set "SOURCE_DIR=C:\Users\KRAFTLAB\Documents\xaubot\MT5_XAUBOT"
set "MT5_DIR=C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5"

REM File paths
set "EA_SOURCE=%SOURCE_DIR%\Experts\XAUUSD_Neural_Bot_v2.mq5"
set "MODEL_SOURCE=%SOURCE_DIR%\Files\lightgbm_real_26features.onnx"

set "EA_TARGET=%MT5_DIR%\Experts\XAUUSD_Neural_Bot_v2.mq5"
set "MODEL_TARGET=%MT5_DIR%\Files\lightgbm_real_26features.onnx"

REM ====================================================================
REM Step 1: Verify source files exist
REM ====================================================================
echo [1/5] Verifying source files...

if not exist "%EA_SOURCE%" (
    echo ERROR: EA file not found!
    echo Expected: %EA_SOURCE%
    echo.
    echo Please ensure you're running this script from the correct location.
    pause
    exit /b 1
)
echo   OK: EA file found

if not exist "%MODEL_SOURCE%" (
    echo ERROR: Model file not found!
    echo Expected: %MODEL_SOURCE%
    echo.
    echo Please ensure the MT5_XAUBOT folder contains the model file.
    pause
    exit /b 1
)
echo   OK: Model file found

REM ====================================================================
REM Step 2: Verify MT5 directories exist
REM ====================================================================
echo.
echo [2/5] Verifying MT5 directories...

if not exist "%MT5_DIR%\Experts" (
    echo WARNING: MT5 Experts directory not found!
    echo Creating: %MT5_DIR%\Experts
    mkdir "%MT5_DIR%\Experts"
)
echo   OK: Experts directory exists

if not exist "%MT5_DIR%\Files" (
    echo WARNING: MT5 Files directory not found!
    echo Creating: %MT5_DIR%\Files
    mkdir "%MT5_DIR%\Files"
)
echo   OK: Files directory exists

REM ====================================================================
REM Step 3: Backup existing files (if any)
REM ====================================================================
echo.
echo [3/5] Checking for existing files...

if exist "%EA_TARGET%" (
    echo   Found existing EA - Creating backup...
    copy /Y "%EA_TARGET%" "%EA_TARGET%.backup" >nul
    echo   OK: Backup created: XAUUSD_Neural_Bot_v2.mq5.backup
) else (
    echo   No existing EA file - Fresh installation
)

if exist "%MODEL_TARGET%" (
    echo   Found existing model - Creating backup...
    copy /Y "%MODEL_TARGET%" "%MODEL_TARGET%.backup" >nul
    echo   OK: Backup created: lightgbm_real_26features.onnx.backup
) else (
    echo   No existing model file - Fresh installation
)

REM ====================================================================
REM Step 4: Copy files
REM ====================================================================
echo.
echo [4/5] Copying files to MT5...

echo   Copying EA...
copy /Y "%EA_SOURCE%" "%EA_TARGET%" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy EA file!
    pause
    exit /b 1
)
echo   OK: EA copied successfully

echo   Copying Model...
copy /Y "%MODEL_SOURCE%" "%MODEL_TARGET%" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy model file!
    pause
    exit /b 1
)
echo   OK: Model copied successfully

REM ====================================================================
REM Step 5: Verify installation
REM ====================================================================
echo.
echo [5/5] Verifying installation...

if exist "%EA_TARGET%" (
    for %%A in ("%EA_TARGET%") do set EA_SIZE=%%~zA
    echo   OK: EA installed - Size: %EA_SIZE% bytes
) else (
    echo ERROR: EA verification failed!
    pause
    exit /b 1
)

if exist "%MODEL_TARGET%" (
    for %%A in ("%MODEL_TARGET%") do set MODEL_SIZE=%%~zA
    echo   OK: Model installed - Size: %MODEL_SIZE% bytes
) else (
    echo ERROR: Model verification failed!
    pause
    exit /b 1
)

REM ====================================================================
REM Success!
REM ====================================================================
echo.
echo ====================================================================
echo  DEPLOYMENT SUCCESSFUL!
echo ====================================================================
echo.
echo Files copied to MT5:
echo   EA:    %EA_TARGET%
echo   Model: %MODEL_TARGET%
echo.
echo ====================================================================
echo  NEXT STEPS:
echo ====================================================================
echo.
echo 1. Open MetaEditor (F4 in MT5)
echo 2. Navigate to: Experts ^> XAUUSD_Neural_Bot_v2.mq5
echo 3. Compile (F7) - Should show "0 errors, 0 warnings"
echo 4. Run backtest in Strategy Tester (Ctrl+R)
echo.
echo Configuration:
echo   Symbol:     XAUUSD
echo   Timeframe:  M1 (1 minute)
echo   Dates:      2022.01.01 - 2024.12.31
echo   Model:      Every tick
echo   Deposit:    10,000
echo.
echo Parameters:
echo   InpConfidenceThreshold: 0.35
echo   InpUseValidation:       false  (CRITICAL!)
echo   InpATRMultiplierSL:     1.5
echo   InpRiskRewardRatio:     2.0
echo.
echo Expected Results: +$45K-$50K profit (+450-500%%)
echo.
echo ====================================================================
echo.
pause
