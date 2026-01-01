# ====================================================================
# Deploy ONNX Model to MT5 Strategy Tester (PowerShell Version)
# ====================================================================

Write-Host "`n========================================"
Write-Host "MT5 Strategy Tester Model Deployment"
Write-Host "========================================`n"

# Define paths
$ProjectDir = $PSScriptRoot
$ModelFile = "lightgbm_xauusd.onnx"
$SourcePath = Join-Path $ProjectDir "MT5_XAUBOT\Files\$ModelFile"
$TesterBase = "C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files"
$DestPath = Join-Path $TesterBase $ModelFile

Write-Host "Source: $SourcePath"
Write-Host "Destination: $DestPath`n"

# Check if source file exists
if (-not (Test-Path $SourcePath)) {
    Write-Host "ERROR: Source file not found!" -ForegroundColor Red
    Write-Host "Looking for: $SourcePath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/3] Checking source file..." -ForegroundColor Cyan
$SourceFile = Get-Item $SourcePath
Write-Host "       Size: $($SourceFile.Length) bytes" -ForegroundColor Green
Write-Host "       OK" -ForegroundColor Green

# Create destination directory
Write-Host "`n[2/3] Creating Tester directory..." -ForegroundColor Cyan
try {
    if (-not (Test-Path $TesterBase)) {
        New-Item -ItemType Directory -Path $TesterBase -Force | Out-Null
        Write-Host "       Created: $TesterBase" -ForegroundColor Green
    } else {
        Write-Host "       Already exists" -ForegroundColor Green
    }
} catch {
    Write-Host "ERROR: Failed to create directory" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Copy model file
Write-Host "`n[3/3] Copying model file..." -ForegroundColor Cyan
try {
    Copy-Item -Path $SourcePath -Destination $DestPath -Force
    Write-Host "       SUCCESS!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Copy failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify copy
if (Test-Path $DestPath) {
    $DestFile = Get-Item $DestPath
    Write-Host "`nVerification:" -ForegroundColor Cyan
    Write-Host "  File: $ModelFile" -ForegroundColor Green
    Write-Host "  Location: $TesterBase" -ForegroundColor Green
    Write-Host "  Size: $($DestFile.Length) bytes" -ForegroundColor Green
    Write-Host "  Date: $($DestFile.LastWriteTime)" -ForegroundColor Green

    Write-Host "`n========================================"
    Write-Host "Deployment Complete!" -ForegroundColor Green
    Write-Host "========================================`n"

    Write-Host "Next steps:"
    Write-Host "1. Open MT5 MetaEditor"
    Write-Host "2. Compile XAUUSD_Neural_Bot_FIXED.mq5"
    Write-Host "3. Open Strategy Tester"
    Write-Host "4. Select XAUUSD_Neural_Bot_FIXED"
    Write-Host "5. Run backtest"
    Write-Host "`nThe model should now load successfully!"
    Write-Host "========================================`n"
} else {
    Write-Host "ERROR: File was not copied successfully" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
