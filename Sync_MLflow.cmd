@echo off
cd /d "%~dp0"
".mlflow-env\Scripts\python.exe" -X utf8 scripts\mlflow_sync.py
pause
