@echo off
set LLM_QUERY_URL=http://localhost:8001/query
set LLM_QUERY_ASYNC_URL=http://localhost:8001/query-async
set LLM_EMBEDDING_URL=http://localhost:8000/get-embedding
set QDRANT_URL=http://localhost:6333
set EMBEDDING_MODEL_PATH=C:/data/Projects/test/instructor-xl
set QUERY_MODEL_PATH=Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_1.bin
start "" uvicorn llm-embedding:app --port 8000
start "" uvicorn llm-query:app --port 8001
start "" uvicorn control:app --port 8002