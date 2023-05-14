##https://www.youtube.com/watch?v=-BidzsQYZM4
##https://www.youtube.com/watch?v=ITV1wv5HiX4

import json
import copy 
from llama_cpp import Llama
import asyncio
import requests
from fastapi import FastAPI, Request, Body
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

## load the model
print("Loading model...")
llm = Llama(model_path="GPT4All-13B-snoozy.ggml.q4_2.bin")   
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
class Question(BaseModel):
    question: str

# question/answer endpoint 
@app.post("/model/")
async def get_answer(question: Question):
  stream = llm(
        "Question:" + question.question + " Answer:",
        max_tokens=200,
        stop=["\n", " Q:", " Question:"],
        echo=False,
        )
  result = copy.deepcopy(stream)  
  text = result["choices"][0]["text"]
  #usage = result["usage"]
  print(text)
  return {"answer" : text}




#async def model(request: Request, question: dict = Body(...)):
#    stream = llm(
#        f"Question: {question['text']} Answer:",
#        max_tokens=100,
#        stop=["\n", " Q:"],
#        stream=True,
#        )
    
#    async def async_generator():
#        for item in stream:
#            yield item
            
#    async def server_sent_events():
#        async for item in async_generator():
#            if await request.is_disconnected():
#                break

#            result = copy.deepcopy(item)
#            text = result["choices"][0]["text"]

#            yield {"data": text}

#    return EventSourceResponse(server_sent_events())














