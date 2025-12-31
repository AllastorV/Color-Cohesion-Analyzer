@echo off
title Color Cohesion Analyzer - Kurulum
cd /d "%~dp0"

echo ========================================
echo   Color Cohesion Analyzer - Kurulum
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

echo [INFO] Python bulundu:
python --version
echo.

:: Create virtual environment
echo [INFO] Virtual environment olusturuluyor...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] pip guncelleniyor...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo [INFO] Gerekli paketler yukleniyor...
echo Bu islem birka dakika surebilir...
echo.
pip install -r requirements.txt

:: Check installation
echo.
echo [INFO] Kurulum kontrol ediliyor...
python -c "import PyQt6; import numpy; import cv2; print('Temel paketler basariyla yuklendi!')"

if errorlevel 1 (
    echo.
    echo [WARNING] Bazi paketler eksik olabilir!
    echo Manuel olarak yuklemek icin: pip install -r requirements.txt
) else (
    echo.
    echo ========================================
    echo   Kurulum Tamamlandi!
    echo ========================================
    echo.
    echo Uygulamayi baslatmak icin: run.bat
)

echo.
pause
