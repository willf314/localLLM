# Query Test Harness

import copy
from llama_cpp import Llama
import sys

print("")
print("#######################################################")
print("")
print("Starting Query Test Client - Asynchronous Calls to LLM")
print("")
print("#######################################################")
print("")

print("Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_1.bin...")
llm = Llama(model_path="Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_1.bin", embedding=False, n_ctx=2048)
print("Model loaded")

# global stream
stream = None

# global function
def generator():
    global stream
    for item in stream:
        answer = item["choices"][0]["text"]
        print(answer,end='')
        sys.stdout.flush()
        yield item

while True:
    # get user input
    question = input(">> ")
    if question.lower() == "exit":
        break
    else:
        # form prompt
        prompt = "Question: " + question + " Answer: "
        # call LLM
        stream = llm(
            prompt,
            max_tokens=1500,
            stop=[" Q:", " Question:"],
            stream=True,
        )
        for _ in generator():
            pass  # Do nothing, just iterate over the generator
















