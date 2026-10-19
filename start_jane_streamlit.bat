@echo off
rem Jane - local AI data analyst (made by Kumar Harsh)
rem The ORIGINAL Streamlit UI on http://localhost:8501.
rem The React app is the default now - launch it with start_jane.bat.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run src\jane\app.py --server.port 8501
pause
