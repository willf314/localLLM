@echo off

start "" uvicorn llm-embedding:app --port 8000 --reload
start "" uvicorn llm-query:app --port 8001 --reload
start "" uvicorn control:app --port 8002 --reload