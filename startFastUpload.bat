@echo off
set QDRANT_URL=http://localhost:6333
set EMBEDDING_MODEL_PATH=C:/data/Projects/test/instructor-xl
start "" uvicorn fast-upload:app --port 8003