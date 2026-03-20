@echo off
echo Pornire Telegram Monitor...

if not exist .venv (
    echo Creez mediu virtual...
    python -m venv .venv
    echo Instalez dependente...
    .venv\Scripts\pip install -r requirements.txt
)

echo Activez mediu virtual...
call .venv\Scripts\activate.bat

echo Verific dependente...
.venv\Scripts\pip show bertopic >nul 2>&1
if errorlevel 1 (
    echo Instalez BERTopic...
    .venv\Scripts\pip install bertopic
)

echo Prima pornire poate dura 2-5 minute (descarcare modele)
start http://localhost:8000
python main.py