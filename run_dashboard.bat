@echo off
cd /d C:\Users\HP\forex-analysis-system
call forex-env\Scripts\activate.bat
streamlit run dashboard/app.py
pause
