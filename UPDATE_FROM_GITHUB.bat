@echo off
REM ====================================================================
REM  Update XAUBOT Project from GitHub
REM  Pulls latest changes to local repository
REM  Created: 2024-12-28
REM ====================================================================

echo.
echo ====================================================================
echo  XAUBOT - Update from GitHub
echo  Syncing local project with latest repository changes...
echo ====================================================================
echo.

REM Navigate to project directory
cd /d "C:\Users\KRAFTLAB\Documents\xaubot"

REM Check git status first
echo [1/3] Checking current status...
git status
echo.

REM Pull latest changes
echo [2/3] Pulling latest changes from GitHub...
git pull origin claude/mt5-model-research-yfrWh
echo.

REM Verify update
echo [3/3] Verifying update...
git log -1 --oneline
echo.

echo ====================================================================
echo  UPDATE COMPLETE!
echo ====================================================================
echo.
echo Latest changes downloaded to: C:\Users\KRAFTLAB\Documents\xaubot
echo.
echo NEXT STEPS:
echo 1. Run DEPLOY_TO_MT5.bat to copy files to MetaTrader 5
echo 2. Open MetaEditor and compile the EA (F7)
echo 3. Continue your backtest in Strategy Tester
echo.
echo ====================================================================
echo.
pause
