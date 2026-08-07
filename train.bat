@echo off
set PYTHONPATH=%~dp0
%~dp0.venv\Scripts\python.exe %~dp0training/train_risk_model.py
