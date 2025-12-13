@echo off
title AI Tutor System - Environment Setup
echo ========================================================
echo       installing AI Tutor System
echo ========================================================
echo.

REM 檢查是否有安裝 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    pause
    exit /b
)

python setup.py

echo.
echo ========================================================
echo       END OF INSTALLATION
echo ========================================================
pause