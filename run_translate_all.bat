@echo off
chcp 65001 >nul
title Auto Manhwa Persian Translator (High-Precision Engine v4)

cd /d "%~dp0"

set "SRC=%~dp0manhwa"
set "OUT=%~dp0manhwa\_translated_fa"
set "WORK=%~dp0manhwa\_work_pro"

set "PY_EXE=py -3.13"
if exist "C:\Users\amir\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PY_EXE=C:\Users\amir\AppData\Local\Programs\Python\Python313\python.exe"
)

rem ============================================================
rem  Optional knobs (uncomment to use):
rem    --langs en,ko     also OCR Korean SFX (needs model download)
rem    --cpu             force CPU if GPU/CUDA is unavailable
rem    --slant 0         upright lettering instead of slight italic
rem    --no-artwork     revert to the old flat-fill behaviour
rem    set MANHWA_API_KEY=...   avoid editing the key into the file
rem ============================================================

echo =================================================================
echo   STARTING FULL MANHWA TRANSLATION TO PERSIAN PDF AND CBZ
echo   Source Folder: "%SRC%"
echo   Output Folder: "%OUT%"
echo =================================================================
echo.

"%PY_EXE%" "%~dp0translate_manhwa_pro.py" "%SRC%" -o "%OUT%" --work "%WORK%"

echo.
echo =================================================================
echo   ALL CHAPTERS TRANSLATED AND SAVED TO PDF!
echo   Location: "%OUT%"
echo =================================================================
pause
