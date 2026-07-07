@echo off
cd /d %~dp0
echo Starting BNP Paribas Cardif Claims Management Platform...
echo.
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
echo.
echo Starting API server on port 8000...
start cmd /k "uvicorn app:app --reload --port 8000"
cd ../frontend
pip install -r requirements.txt
timeout /t 5
echo.
echo Starting Streamlit UI on port 8501...
start cmd /k "streamlit run app.py"
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo API Docs: http://localhost:8000/docs
