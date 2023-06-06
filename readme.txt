Document Chat Services

Installation

Files:
1.control.py : AI Control Service
2.llm-embedding.py: LLM configured to generate embeddings
3.llm-query.py: LLM configured for text completion (answering queries)
4.fast-upload.py: Control service combined with embedding LLM to speed up document ingestion
5. startAI.bat: batch file to set env variables and start web-services, one service for each of the scripts above 
6.startFastUpload.bat: batch file to start fast upload service
7.client.html: test client with javascript to connect to the services

Each service produces a log file, same content is also streamed to stdout:
1.control.log
2.llm-embedding.log
3.llm-query.log

Install requirements:
1.pip install fastapi
2.pip install uvicorn[standard]
3.make sure python/scripts folder is in the path for uvicorn
4.run uvicorn main:app --reload to start the server process
5.pip install ayncio
6.pip install sse-starlette
7.pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
8.for embedding LLM: pip install InstructorEmbedding
9.pip install langchain --force-reinstall --upgrade --no-cache-dir

Qdrant vector DB must be setup separately. Once the db is started, update startAI.bat with the db URL and port to connect on 

Original query LLM model can be found here:
https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/tree/previous_llama
Download this file and put it in the working directory alongside python scripts.

Now using:
TheBloke/Wizard-Vicuna-7B-Uncensored-GGML

The embedding LLM:
https://huggingface.co/hkunlp/instructor-xl

use this command to download a local instance:
git lfs clone https://huggingface.co/hkunlp/instructor-xl

Then set the EMBEDDING_MODEL_PATH environment variable to point at the download folder location. 

Note: use forward slashes for embedding model path. It failed to load on windows, using backslashes.

The alternative is to use the default hugging face code to automatically download. This code is included in the llm-embedding python file but commented out. It would automatically check for updates and download new versions to the ./cache location everytime the llm-embedding service is started. This could lead to a substantial delay in service start up time.

Start up:

set webservice endpoint URLs env variables in startAI.bat file
set startup ports in uvicorn commands in startAI.bat
repeat same config for startFastUpload.bat
run startAI.bat to start-up the query services - embedding llm, query llm and control service
run startFastUpload.bat to start the fast upload service
the command executed for each service is: uvicorn [python file name]:app --reload --port [port to run on]
this starts the webservices using the Python FASTAPI framework

for unix set the following in etc/environment
LLM_QUERY_URL=http://localhost:8001/query
LLM_QUERY_ASYNC_URL=http://localhost:8001/query-async
LLM_EMBEDDING_URL=http://localhost:8000/get-embedding
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL_PATH=C:/data/Projects/test/instructor-xl
QUERY_MODEL_PATH=Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_1.bin

API Info

AI Control Service (control.py)

1. /ingest-pdf  POST
Takes a binary file as input (must be PDF), converts it to text, splits the text into chunks, gets an embedding for each chunk, and persists the embedding and chunk into the Vector DB.

Javascript to post the PDF file as binary data:
var reader = new FileReader();
reader.onload = function (event) {
var fileContent = event.target.result;
var fileBlob = new Blob([fileContent], { type: selectedFile.type });
var formData = new FormData();                    
formData.append('file', fileBlob, selectedFile.name);
...
data: formData
...
}

2. /query-docs POST 
Accepts requests in JSON format: { "text": question }
Takes a question as input, retrieves the embedding for the question, retrieves matching chunks from the Vector DB, forms a prompt including the chunks as context and the question, sends prompt to the query LLM and returns the answer.

Control service needs Vector DB API endpoints equivalent to:

1. file_exists_in_vectorDB(filename)
returns true if any chunks with this filename already exist in the DB

2. persist_to_vectorDB(filename, text_chunk, embedding)
Embedding will be an array of 5020 floating point numbers (the embedding vector)
[0.123123 , 0.34324234, 0.34232423 ...]

3. retrieve_matches_from_vectorDB(embedding,max_chunks)
Given a vector array as input, returns back an array of text chunks
["chunk 1...", "chunk 2...", "chunk 3...", ...]
Returns up to max_chunks if it can find reasonable matches with a similarity search.











