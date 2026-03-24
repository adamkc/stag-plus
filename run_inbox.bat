@echo off
echo ============================================
echo  STAG+ — Processing _Inbox
echo ============================================
echo.

set VENV=%~dp0.venv\Scripts\python.exe
set STAG=%~dp0stag.py
set INBOX=Z:\PhotoEdits\_Inbox

if not exist "%INBOX%" (
    echo ERROR: Inbox folder not found: %INBOX%
    pause
    exit /b 1
)

echo Directory: %INBOX%
echo Features:  IQ + Aesthetics (no RAM+ tagging)
echo Mode:      darktable-compatible filenames
echo.

"%VENV%" "%STAG%" "%INBOX%" --no-tags --iq --aes --prefer-exact-filenames

echo.
echo ============================================
echo  Done! You can now import in darktable.
echo ============================================
pause
