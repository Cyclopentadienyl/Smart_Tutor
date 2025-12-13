@echo off
REM Force change to script directory
cd /d "%~dp0"

title AI Tutor System - Starting
echo ========================================================
echo       Launching AI Tutor System
echo ========================================================
echo Current Directory: %cd%
echo.

REM --- Check virtual environment (using GOTO to avoid syntax errors) ---
if exist ".venv\Scripts\python.exe" goto StartApp

REM --- If environment not found, jump here ---
:ErrorMissing
echo [ERROR] Virtual environment not found (.venv\Scripts\python.exe)!
echo [CHECK] 1. Please make sure you have run 'install.bat' and saw the success message.
echo [CHECK] 2. Please verify your folder structure is correct.
echo.
pause
exit /b

REM --- Start application ---
:StartApp
echo Calling virtual environment Python...
".venv\Scripts\python.exe" launch.py

REM --- End handling ---
if %errorlevel% neq 0 goto ErrorRun
goto End

:ErrorRun
echo.
echo [WARNING] Program terminated abnormally (Error Level: %errorlevel%)
echo Please check the error messages above.

:End
pause