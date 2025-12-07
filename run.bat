@echo off
REM 強制切換到腳本所在目錄
cd /d "%~dp0"

REM 設定編碼為 UTF-8
chcp 65001 >nul

title AI 家教系統 - 啟動中
echo ========================================================
echo       正在啟動 AI 家教系統 (Launch)
echo ========================================================
echo 目前目錄: %cd%
echo.

REM --- 檢查虛擬環境 (使用 GOTO 避免括號語法錯誤) ---
if exist ".venv\Scripts\python.exe" goto StartApp

REM --- 如果找不到環境，跳轉到這裡 ---
:ErrorMissing
echo [錯誤] 找不到虛擬環境 (.venv\Scripts\python.exe)！
echo [檢查] 1. 請確認您是否已執行 'install.bat' 且看到成功訊息。
echo [檢查] 2. 請確認您的資料夾結構是否正確。
echo.
pause
exit /b

REM --- 啟動應用程式 ---
:StartApp
echo 正在呼叫虛擬環境 Python...
".venv\Scripts\python.exe" launch.py

REM --- 結束處理 ---
if %errorlevel% neq 0 goto ErrorRun
goto End

:ErrorRun
echo.
echo [警告] 程式異常結束 (Error Level: %errorlevel%)
echo 請檢查上方的錯誤訊息。

:End
pause