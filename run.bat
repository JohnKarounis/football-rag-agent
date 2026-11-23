@echo off
cd /d %~dp0
start "API" cmd /k python -m uvicorn football_rag.api.main:app --reload --env-file .env
start "UI" cmd /k python -m football_rag.gradio_ui