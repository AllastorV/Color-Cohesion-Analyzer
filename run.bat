@echo off
title Color Cohesion Analyzer
cd /d "%~dp0"

echo ========================================
echo   Color Cohesion Analyzer
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python bulunamadi!
    echo Python 3.9 veya uzeri yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment aktif ediliyor...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Virtual environment bulunamadi, sistem Python kullaniliyor...
)

:: Run the application
echo [INFO] Uygulama baslatiliyor...
echo.
python main.py

:: If there's an error, pause to show it
if errorlevel 1 (
    echo.
    echo [ERROR] Uygulama hata ile sonlandi!
    pause
)
