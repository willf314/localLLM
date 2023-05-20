
import json
import copy 
from llama_cpp import Llama
import asyncio
#import requests
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

## load the model
print("\n###################################")
print("")
print("Starting llm-query")
print("")
print("###################################\n")

print("Loading model...")
llm = Llama(model_path="GPT4All-13B-snoozy.ggml.q4_2.bin", embedding=False, n_ctx = 4096)   
print("Model loaded")

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
    
    print("running sync query:" + str(request.text) + "\n")

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

    #log raw result to console
    print(result)
    
    #return result to client
    return {"question" : request.text , "answer" : text, "id" : id, "object" : object,
           "created" : created, "modelName" : modelName, 
           "prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens,
           "total_tokens" : total_tokens}


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
    print("running async query:" + str(request.text) + "\n")
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
        
    async def server_sent_events():
        async for item in async_generator():
            if await request.is_disconnected():
                break

            result = copy.deepcopy(item)
            text = result["choices"][0]["text"]                       
            yield {"data": text}
            
    return EventSourceResponse(server_sent_events())
















