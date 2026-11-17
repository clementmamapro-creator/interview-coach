@echo off
cd /d %~dp0
where python >nul 2>nul || (echo Python introuvable : installez Python 3.12 depuis python.org en cochant "Add python.exe to PATH" & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Installation terminee. Collez vos cles dans le fichier .env qui va s ouvrir, enregistrez, puis lancez start.bat
notepad .env
pause
