@echo off
setlocal
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment. Check that Python 3.11 is installed.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Review the error shown above.
  pause
  exit /b 1
)
echo Setup complete. Activate with: .venv\Scripts\activate
pause
