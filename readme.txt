Document Chat Services

Installation

Files:
1.control.py : AI Control Service
2.llm-embedding: LLM configured to generate embeddings
3.llm-query: LLM configured for text completion (answering queries)
4.startAI.bat: batch file to start the above components on ports 8000, 8001, 8002
5.GPT4All-13B-snoozy.ggml.q4_2.bin: the LLM (same LLM currently used for both services) 
6. client.html: test client with javascript to connect to the services

Each service produces a log file, same content is also streamed to stdout:
1. control.log
2. llm-embedding.log
3. llm-query.log

There is no config file, all parameters are hardcoded in the python scripts

Install requirements:
1.pip install fastapi
2.pip install uvicorn[standard]
3.make sure python/scripts folder is in the path for uvicorn
4.run uvicorn main:app --reload to start the server process
5.pip install ayncio
6.pip install sse-starlette
7.Install llama cpp + the python library

The LLM can be sourced from:
https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/tree/previous_llama

Start up:
run startAI.bat to start-up the services
the command executed for each service is: uvicorn [python file name]:app --reload --port [port to run on]
this starts the webservices using the Python FASTAPI framework

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











