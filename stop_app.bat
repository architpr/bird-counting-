@echo off
echo Stopping all Python processes...
taskkill /F /IM python.exe
taskkill /F /IM uvicorn.exe
taskkill /F /IM streamlit.exe
echo All processes stopped.
pause
