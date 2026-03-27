@echo off
echo.
echo ============================================
echo  STAG+ Inbox Processor
echo  Score + Sort + Deduplicate
echo ============================================
echo.

set VENV=%~dp0.venv\Scripts\python.exe
set SCRIPT=%~dp0process_inbox.py

echo Step 1: Dry run (preview)...
echo.
"%VENV%" "%SCRIPT%"

echo.
echo ============================================
echo  Ready to execute? This will:
echo    1. Run STAG+ (tags, IQ, aesthetics)
echo    2. Move files to date folders
echo    3. Skip duplicates
echo ============================================
echo.
set /p CONFIRM=Type YES to proceed:
if /i not "%CONFIRM%"=="YES" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
"%VENV%" "%SCRIPT%" --execute

echo.
pause
