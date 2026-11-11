@echo off
cd /d "%~dp0"
".mlflow-env\Scripts\python.exe" -X utf8 scripts\mlflow_ui.py
if errorlevel 1 pause
