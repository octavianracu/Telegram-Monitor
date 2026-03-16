@echo off
echo ========================================
echo    Telegram Monitor - Instalare
echo ========================================
echo.

REM Verifică dacă Python este instalat
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python NU este instalat!
    echo.
    echo Descarcă Python de la: https://www.python.org/downloads/
    echo Asigură-te că bifezi "Add Python to PATH" la instalare.
    pause
    exit /b 1
)

echo [OK] Python este instalat
echo.

REM Verifică versiunea Python
python -c "import sys; exit(0) if sys.version_info >= (3,8) else exit(1)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python 3.8 sau mai nou este recomandat
    echo Versiunea ta:
    python --version
    echo.
    choice /c YN /m "Continui oricum?"
    if errorlevel 2 exit /b 1
)

echo [OK] Versiune Python compatibila
echo.

REM Creează virtual environment
echo [1/5] Creez virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Nu pot crea virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment creat
echo.

REM Activează virtual environment
echo [2/5] Activez virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Nu pot activa virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activat
echo.

REM Upgrade pip
echo [3/5] Actualizez pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] Pip actualizat
echo.

REM Instalează dependențele
echo [4/5] Instalez dependențele (asta poate dura 5-10 minute)...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Eroare la instalarea dependențelor
    pause
    exit /b 1
)

echo [OK] Dependențe instalate
echo.

REM Creează folderele necesare
echo [5/5] Creez folderele necesare...
mkdir projects >nul 2>&1
mkdir backups >nul 2>&1
mkdir static >nul 2>&1
echo [OK] Foldere create
echo.

echo ========================================
echo    INSTALARE FINALIZATA CU SUCCES!
echo ========================================
echo.
echo Pentru a porni aplicatia:
echo ----------------------------------------
echo 1. Ruleaza: run.bat
echo 2. Deschide browser: http://localhost:8000
echo.
echo Pentru a opri aplicatia: Ctrl+C
echo.

pause