@echo off
chcp 65001 >nul
title AI 家教系統 - 環境部署
echo ========================================================
echo       正在執行 AI 家教系統一鍵部署 (install)
echo ========================================================
echo.

REM 檢查是否有安裝 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 找不到 Python！請確認您已安裝 Python 3.10 以上版本並加入 PATH 環境變數。
    echo 下載連結: https://www.python.org/downloads/
    pause
    exit /b
)

echo 正在呼叫 setup.py 進行安裝...
python setup.py

echo.
echo ========================================================
echo       安裝程序結束
echo ========================================================
pause