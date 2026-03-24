@echo off
echo ============================================
echo  STAG+ — Full Library Scan
echo ============================================
echo.

set VENV=%~dp0.venv\Scripts\python.exe
set STAG=%~dp0stag.py
set PHOTOS=Z:\PhotoEdits

echo Directory: %PHOTOS%
echo Features:  Tags + IQ + Aesthetics (ALL)
echo Mode:      darktable-compatible filenames
echo Note:      Skips already-processed images
echo.
echo This will take a while on first run (13k+ images).
echo Press Ctrl+C to cancel at any time.
echo.
pause

"%VENV%" "%STAG%" "%PHOTOS%" --all --prefer-exact-filenames

echo.
echo ============================================
echo  Done!
echo ============================================
pause
