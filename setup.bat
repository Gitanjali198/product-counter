@echo off
echo Installing requirements...
python -m pip install -r requirements.txt
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Created the file .env - open it and paste your Gemini API key after GEMINI_API_KEY=
)
pause
