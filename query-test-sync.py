# Query Test Harness

import json
import copy 
from llama_cpp import Llama
import asyncio
import sys

# load the model
print("")
print("######################################################")
print("")
print("Starting Query Test Client - Synchronous Calls to LLM")
print("")
print("######################################################")
print("")

print("Loading model GPT4All-13B-snoozy.ggml.q4_2.bin...")
llm = Llama(model_path="GPT4All-13B-snoozy.ggml.q4_2.bin", embedding=False, n_ctx = 2048)
print("Model loaded")

while True:
    # get user input
    question = input(">> ")
    if question.lower() == "exit":
            break
    else:
        #form prompt
        prompt = "Question: " + question + " Answer: "    
    
        #call LLM        
        stream = llm(                
        prompt,
        max_tokens=1500,
        stop=[ " Q:", " Question:"],
        echo=False,
        )        

        #display answer
        result = copy.deepcopy(stream) 
        answer = result["choices"][0]["text"]
        print(answer)















