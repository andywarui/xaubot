# Build LightGBM MT5 DLL
# Run after installing Visual Studio Build Tools

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building LightGBM MT5 DLL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$vcpkgPath = "C:\vcpkg"

# Step 1: Install LightGBM via vcpkg
Write-Host "[1/4] Installing LightGBM via vcpkg..." -ForegroundColor Yellow
Write-Host "  This may take 15-30 minutes..." -ForegroundColor Gray
Write-Host ""

if (Test-Path "$vcpkgPath\installed\x64-windows\include\LightGBM") {
    Write-Host "  ✓ LightGBM already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing LightGBM:x64-windows..." -ForegroundColor Green
    & "$vcpkgPath\vcpkg.exe" install lightgbm:x64-windows

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: LightGBM installation failed!" -ForegroundColor Red
        pause
        exit 1
    }

    Write-Host "  ✓ LightGBM installed successfully" -ForegroundColor Green
}

Write-Host ""

# Step 2: Configure CMake
Write-Host "[2/4] Configuring CMake..." -ForegroundColor Yellow

$buildDir = Join-Path $projectRoot "lightgbm_mt5_dll\build"

if (Test-Path $buildDir) {
    Write-Host "  Cleaning old build directory..." -ForegroundColor Gray
    Remove-Item -Path $buildDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Set-Location $buildDir

Write-Host "  Running CMake configuration..." -ForegroundColor Green

cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE="$vcpkgPath\scripts\buildsystems\vcpkg.cmake"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Write-Host "Make sure Visual Studio Build Tools are installed" -ForegroundColor Yellow
    Set-Location $projectRoot
    pause
    exit 1
}

Write-Host "  ✓ CMake configured successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Build DLL
Write-Host "[3/4] Building DLL..." -ForegroundColor Yellow
Write-Host "  Compiling C++ code..." -ForegroundColor Green
Write-Host ""

cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Set-Location $projectRoot
    pause
    exit 1
}

Write-Host ""
Write-Host "  ✓ DLL built successfully" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy to MT5
Write-Host "[4/4] Deploying to MT5..." -ForegroundColor Yellow

$dllSource = Join-Path $buildDir "bin\Release\lightgbm_mt5.dll"
$modelSource = Join-Path $projectRoot "MT5_XAUBOT\Files\lightgbm_xauusd.txt"

# Find MT5 installation
$mt5Paths = @(
    "$env:APPDATA\MetaQuotes\Terminal",
    "$env:PROGRAMFILES\MetaTrader 5",
    "C:\Program Files\MetaTrader 5"
)

$mt5Terminal = $null
foreach ($path in $mt5Paths) {
    if (Test-Path $path) {
        $terminals = Get-ChildItem -Path $path -Directory -ErrorAction SilentlyContinue
        if ($terminals) {
            $mt5Terminal = $terminals[0].FullName
            break
        }
    }
}

if ($mt5Terminal) {
    $dllDest = Join-Path $mt5Terminal "MQL5\Libraries\lightgbm_mt5.dll"
    $libDllDest = Join-Path $mt5Terminal "MQL5\Libraries\lib_lightgbm.dll"
    $modelDest = Join-Path "$env:APPDATA\MetaQuotes\Terminal\Common\Files" "lightgbm_xauusd.txt"

    # Create directories if needed
    New-Item -ItemType Directory -Force -Path (Split-Path $dllDest) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $modelDest) | Out-Null

    # Copy files to terminal
    Copy-Item -Path $dllSource -Destination $dllDest -Force
    Copy-Item -Path (Join-Path $buildDir "bin\Release\lib_lightgbm.dll") -Destination $libDllDest -Force
    Copy-Item -Path $modelSource -Destination $modelDest -Force

    # Copy vcpkg dependencies (fmt.dll)
    $fmtDll = "$vcpkgPath\installed\x64-windows\bin\fmt.dll"
    if (Test-Path $fmtDll) {
        Copy-Item -Path $fmtDll -Destination (Join-Path $mt5Terminal "MQL5\Libraries\fmt.dll") -Force
    }

    Write-Host "  ✓ DLL copied to: $dllDest" -ForegroundColor Green
    Write-Host "  ✓ lib_lightgbm.dll copied to terminal" -ForegroundColor Green
    Write-Host "  ✓ Model copied to: $modelDest" -ForegroundColor Green

    # Also copy to Strategy Tester agent directories
    $testerPath = "$env:APPDATA\MetaQuotes\Tester"
    if (Test-Path $testerPath) {
        # Find all terminal directories
        $terminalDirs = Get-ChildItem -Path $testerPath -Directory -ErrorAction SilentlyContinue
        foreach ($terminalDir in $terminalDirs) {
            # Find all Agent-* directories within each terminal
            $agentPattern = Join-Path $terminalDir.FullName "Agent-*"
            $agentDirs = Get-ChildItem -Path $terminalDir.FullName -Directory -Filter "Agent-*" -ErrorAction SilentlyContinue

            foreach ($agentDir in $agentDirs) {
                $agentLibPath = Join-Path $agentDir.FullName "MQL5\Libraries"
                $agentFilesPath = Join-Path $agentDir.FullName "MQL5\Files"

                # Create directories if needed
                if (-not (Test-Path $agentLibPath)) {
                    New-Item -ItemType Directory -Force -Path $agentLibPath | Out-Null
                }
                if (-not (Test-Path $agentFilesPath)) {
                    New-Item -ItemType Directory -Force -Path $agentFilesPath | Out-Null
                }

                # Copy DLLs
                Copy-Item -Path $dllSource -Destination (Join-Path $agentLibPath "lightgbm_mt5.dll") -Force
                Copy-Item -Path (Join-Path $buildDir "bin\Release\lib_lightgbm.dll") -Destination (Join-Path $agentLibPath "lib_lightgbm.dll") -Force
                if (Test-Path $fmtDll) {
                    Copy-Item -Path $fmtDll -Destination (Join-Path $agentLibPath "fmt.dll") -Force
                }

                # Copy model file
                Copy-Item -Path $modelSource -Destination (Join-Path $agentFilesPath "lightgbm_xauusd.txt") -Force

                Write-Host "  ✓ All files copied to: $($terminalDir.Name)\$($agentDir.Name)" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "  ! MT5 installation not found" -ForegroundColor Yellow
    Write-Host "  Manually copy:" -ForegroundColor Yellow
    Write-Host "    DLL: $dllSource" -ForegroundColor White
    Write-Host "    To: [MT5]\MQL5\Libraries\" -ForegroundColor White
    Write-Host ""
    Write-Host "    Model: $modelSource" -ForegroundColor White
    Write-Host "    To: Terminal\Common\Files\" -ForegroundColor White
}

Set-Location $projectRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "DLL Location: $dllSource" -ForegroundColor White
Write-Host "DLL Size: $((Get-Item $dllSource).Length / 1KB) KB" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Open MT5 MetaEditor" -ForegroundColor White
Write-Host "2. Open: mt5_expert_advisor\XAUUSD_NeuralBot_DLL.mq5" -ForegroundColor White
Write-Host "3. Compile (F7)" -ForegroundColor White
Write-Host "4. Enable 'Allow DLL imports' in MT5 Options" -ForegroundColor White
Write-Host "5. Run in Strategy Tester!" -ForegroundColor White
Write-Host ""

pause
