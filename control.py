# AI Control Service

import io
import asyncio
import PyPDF2
from fastapi import FastAPI, UploadFile, File
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from langchain.text_splitter import CharacterTextSplitter
import requests

print("\n###################################")
print("")
print("Starting AI control service")
print("")
print("###################################\n")

app = FastAPI()

# Set up CORS middleware - add the Access-Control-Allow-Origin header 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# convert PDF file to text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# retreive embeddings for a chunk of text
def get_embedding(chunk):

    # Define the endpoint URL
    endpoint_url = "http://localhost:8000/get-embedding"

    # Define the request payload
    payload = {
        "text": chunk
    }

    # Make a POST request to the endpoint
    response = requests.post(endpoint_url, json=payload)

    # Check the response status code
    if response.status_code == 200:
        # Get the response JSON
        response_json = response.json()
        embeddings = response_json["embeddings"]
        # Process the embeddings as needed
        print("Embeddings succesfully retrieved")
        return(embeddings)
    else:
        print("Request failed with status code:", response.status_code)
        return()

# persist one chunk in the VectorDB
def persist_to_vectorDB(chunk,embeddings):
    #to do - add code to call Vector DB web service when its ready
    return()

# webservice endpoint to load PDF content into vector DB
@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    print ("/ingest-pdf called")
    # convert PDF file to text
    raw_text = extract_text_from_pdf(io.BytesIO(await file.read()))
    print("converted PDF to text")
    # split text into chunks        
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        )

    texts = text_splitter.split_text(raw_text)
    print("split text into " + str(len(texts)) + " chunks")
    
    # make web-service calls to get embeddings for each chunk, and persist to Vector db
    i=0
    for chunk in texts:        
        i+=1
        embeddings = get_embedding(chunk)
        persist_to_vectorDB(chunk, embeddings)
        print("ingested chunk:"+ str(i) + "[" + chunk + "]\n")
        
    # return result to client    
    return {"message": "PDF file successfully ingested", "chunks" : str(len(texts)), "chunk size" : "1000", 
            "chunk_overlap" : "200", "text": raw_text}


#@app.post("/ask-question")
#@app.post("/ask-question-async")





