@echo off
REM ====================================================================
REM Deploy ONNX Model to MT5 Strategy Tester
REM ====================================================================
echo.
echo ========================================
echo MT5 Strategy Tester Model Deployment
echo ========================================
echo.

REM Define paths
set "PROJECT_DIR=%~dp0"
set "MODEL_FILE=lightgbm_xauusd.onnx"
set "SOURCE=%PROJECT_DIR%MT5_XAUBOT\Files\%MODEL_FILE%"
set "TESTER_BASE=C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files"

echo Source: %SOURCE%
echo Destination: %TESTER_BASE%
echo.

REM Check if source file exists
if not exist "%SOURCE%" (
    echo ERROR: Source file not found!
    echo Looking for: %SOURCE%
    pause
    exit /b 1
)

echo [1/3] Checking source file...
for %%A in ("%SOURCE%") do echo        Size: %%~zA bytes
echo        OK

REM Create destination directory
echo.
echo [2/3] Creating Tester directory...
if not exist "%TESTER_BASE%" (
    mkdir "%TESTER_BASE%"
    if errorlevel 1 (
        echo ERROR: Failed to create directory
        pause
        exit /b 1
    )
    echo        Created: %TESTER_BASE%
) else (
    echo        Already exists
)

REM Copy model file
echo.
echo [3/3] Copying model file...
copy /Y "%SOURCE%" "%TESTER_BASE%\%MODEL_FILE%"
if errorlevel 1 (
    echo ERROR: Copy failed!
    pause
    exit /b 1
)

REM Verify copy
if exist "%TESTER_BASE%\%MODEL_FILE%" (
    echo        SUCCESS!
    echo.
    for %%A in ("%TESTER_BASE%\%MODEL_FILE%") do (
        echo Verification:
        echo   File: %MODEL_FILE%
        echo   Location: %TESTER_BASE%
        echo   Size: %%~zA bytes
        echo   Date: %%~tA
    )
    echo.
    echo ========================================
    echo Deployment Complete!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Open MT5 MetaEditor
    echo 2. Compile XAUUSD_Neural_Bot_FIXED.mq5
    echo 3. Open Strategy Tester
    echo 4. Select XAUUSD_Neural_Bot_FIXED
    echo 5. Run backtest
    echo.
    echo The model should now load successfully!
    echo ========================================
) else (
    echo ERROR: File was not copied successfully
    pause
    exit /b 1
)

echo.
pause
