# Large Language Model Service configured to produce embeddings (llm-embedding.py)

import json
import copy 
from llama_cpp import Llama
import asyncio
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
import logging
from logging.handlers import RotatingFileHandler
import sys

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

logger.info("Loading model...")
llm = Llama(model_path="GPT4All-13B-snoozy.ggml.q4_2.bin", embedding=True, n_ctx = 2048)   


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

# question/answer endpoint - synchronous - blocks until LLM returns full result
@app.post("/query")
async def query(request: LLMRequest):
    
    logger.info("/query API called")
    logger.info("Query:[%s]", request.text)
    logger.info("")

    #call LLM
    stream = llm(        
        #f"Question:{request.text} Answer:",
        request.text,
        max_tokens=2048,
        stop=[ " Q:", " Question:"],
        echo=False,
        )
    
    #retrieve answer + supporting data from the LLM result
    result = copy.deepcopy(stream)  
    text = result["choices"][0]["text"]
    id = result["id"]
    object = result["object"]
    created = result["created"]
    modelName = result["model"]
    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    total_tokens = result["usage"]["total_tokens"]

    #log results 
    logger.info("llm response:[%s]",text)
    logger.info("id:[" + id + "]")
    logger.info("model:[" + modelName + "]")
    logger.info("prompt_tokens:[" + str(prompt_tokens) + "]")
    logger.info("completion_tokens:[" + str(prompt_tokens) + "]")
    logger.info("prompt_tokens:[" + str(total_tokens) + "]")
    
    #return result to client
    return {"question" : request.text , "answer" : text, "id" : id, "object" : object,
           "created" : created, "modelName" : modelName, 
           "prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens,
           "total_tokens" : total_tokens}


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
    embedding = llm.embed(request.text)        
    enumerateEmbedding(embedding)

    #log result    
    logger.info("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")    
    logger.info("completed embedding request")
    logger.info("")
    
    #return result
    return {"embedding" : embedding}


# question/answer endpoint - asynchronous 

#global stream
stream = None

#global function
async def async_generator():
        for item in stream:
            yield item

@app.post("/query-async")
async def query_async(request: LLMRequest):                
    #call LLM    
    logger.info("/query-async API called")
    logger.info("Query:[%s]",request.text)
    global stream
    stream = llm(
        #f"Question:{request.text} Answer:",
        request.text,
        max_tokens=2048,
        stop=[ " Q:", "Question:"],
        stream=True,
        )
        
    return {}

# stream back LLM response to client
@app.get("/stream-answer")
async def stream_answer(request: Request):
    global stream 
    logger.debug("/stream-answer API called")        
    async def server_sent_events():
        async for item in async_generator():
            if await request.is_disconnected():
                break

            result = copy.deepcopy(item)
            text = result["choices"][0]["text"]                       
            logger.debug("streaming data:[%s]",text)
            yield {"data": text}
            
    return EventSourceResponse(server_sent_events())
















