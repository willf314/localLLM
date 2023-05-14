llm from hugging face:
https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/tree/previous_llama
"GPT4All-13B-snoozy.ggml.q4_2.bin" 7,946,066 KB

install and setup notes:
1)pip install fastapi
2)pip install uvicorn[standard]
3)make sure python/scripts folder is in the path for uvicorn
4)run uvicorn main:app --reload to start the server process
5)pip install ayncio
6)pip install sse-starlette

