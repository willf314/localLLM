@echo off
set LLM_QUERY_URL=http://localhost:8001/query
set LLM_QUERY_ASYNC_URL=http://localhost:8001/query-async
set LLM_EMBEDDING_URL=http://localhost:8000/get-embedding
start "" uvicorn llm-embedding:app --port 8000 --reload
start "" uvicorn llm-query:app --port 8001 --reload
start "" uvicorn control:app --port 8002 --reload