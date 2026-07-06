@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo No se encontro Python en PATH. Instala Python 3.11 o superior y vuelve a ejecutar este instalador.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual .venv...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Actualizando pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist input mkdir input
if not exist output mkdir output
if not exist output\cropped mkdir output\cropped
if not exist output\debug mkdir output\debug
if not exist output\needs_review mkdir output\needs_review

echo.
echo Instalacion finalizada.
echo Copia tus fotos en input y ejecuta run.bat.
exit /b 0
