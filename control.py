# AI Control Service

import io
import PyPDF2
from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    
    # convert PDF file to text
    text = extract_text_from_pdf(io.BytesIO(await file.read()))
    
    # split text into chunks
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        )

    texts = text_splitter.split_text(raw_text)

    # make web-service calls to get embeddings for each chunk, and persist to Vector db
    for chunk in texts:
        embeddings = get_embedding(chunk)
        persist_to_vectorDB(chunk)
    
    # return result to client    
    return {"message": "PDF file successfully ingested", "chunks" : str(len(texts)), "chunk size" : chunksize, 
            "chunk_overlap" : chunkoverlap, "text": text}

@app.post("/ask-question")




@app.post("/ask-question-async")





