@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No existe .venv. Ejecutando install.bat...
    call install.bat
    if errorlevel 1 exit /b 1
)

if not exist input mkdir input
if not exist output mkdir output

".venv\Scripts\python.exe" main.py --input input --output output %*
exit /b %ERRORLEVEL%
