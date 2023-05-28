# Large Language Model Service configured to produce embeddings (llm-embedding.py)

import json
import copy 
#from llama_cpp import Llama  # swapping this out for the instructor model below
from InstructorEmbedding import INSTRUCTOR
import asyncio
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
import logging
from logging.handlers import RotatingFileHandler
import sys
import os 
from typing import List
import numpy as np

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = RotatingFileHandler('llm-embedding.log', maxBytes=10485760, backupCount=5, encoding='utf-8')   # rotate after 5x10MB log files
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

## load the model
logger.info("")
logger.info("###################################")
logger.info("")
logger.info("Starting llm-embedding")
logger.info("")
logger.info("###################################")
logger.info("")

# load environment variables
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")


logger.info("Loading model from " + EMBEDDING_MODEL_PATH + "...")
#llm = Llama(model_path="GPT4All-13B-snoozy.ggml.q4_2.bin", embedding=True, n_ctx = 2048)   
llm = INSTRUCTOR(EMBEDDING_MODEL_PATH)
logger.info("Model loaded")

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

#define the request body schema
class LLMRequest(BaseModel):
    text: str

# helper function to trim the length of a chunk, and remove any newline characters to it prints better
def trimChunk(chunk, max_length):
    chunk = chunk.replace('\n', ' ').replace('\r', ' ')
    if len(chunk) <= max_length:
        return chunk
    else:
        return chunk[:max_length] + "..."

# helper function to trim the length of embedding
def trimEmbedding(embedding,max_length):
    str_repr = ", ".join([str(num) for num in embedding])
    if len(str_repr) <= max_length:
        return str_repr
    else:
        trimmed_str = str_repr[:max_length] 
        return trimmed_str + " ..."

# helper function to count and log individual embedding
def enumerateEmbedding(embedding):
    for index, number in enumerate(embedding):
        logger.debug("Index:[" + str(index) + "] Number:[" + str(number) + "]")

# API to create embedding from a text string
@app.post("/get-embedding")
async def get_embedding(request: LLMRequest):
    # log request
    logger.info("/get-embedding API called")    
    logger.info("chunk size:" + str(len(request.text)))
    logger.info("text:[%s]", trimChunk(request.text,80))
    logger.info("retreiving embedding...")
    
    # retrieve embedding and log individual numbers
    #embedding = llm.embed(request.text)        # use this code for llama cpp models
    embedding = llm.encode(request.text)         # use this code for the hugging face instructor XL model
    enumerateEmbedding(embedding)
   
    #log result    
    logger.info("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")    
    logger.info("completed embedding request")
    logger.info("")
    
    #return result
    embedding_list = np.array(embedding).tolist()
    return {"embedding" : embedding_list}

















