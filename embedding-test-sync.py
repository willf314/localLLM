# Query Test Harness

import json
import copy 
from InstructorEmbedding import INSTRUCTOR
import asyncio
import sys

# load the model
print("")
print("######################################################")
print("")
print("Starting Embedding Test Client - Synchronous Calls to LLM")
print("")
print("######################################################")
print("")

print('Loading model C:/data/Projects/test/instructor-xl...')
llm = INSTRUCTOR('C:/data/Projects/test/instructor-xl')
print("Model loaded")

while True:
    # get user input
    question = input(">> ")
    if question.lower() == "exit":
            break
    else:               
        #call LLM        
        embedding = llm.encode(question)           

        #display result
        for index, number in enumerate(embedding):
            print("Index:[" + str(index) + "] Number:[" + str(number) + "]")

        















