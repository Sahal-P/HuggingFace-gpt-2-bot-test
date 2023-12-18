from typing import Union
import os
from fastapi import FastAPI, Query
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.responses import JSONResponse
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from fastapi.middleware.cors import CORSMiddleware

import requests


# pip install transformers
# huggingface_hub
# sentence_transformers
app = FastAPI()


# Configure CORS
origins = [
    "http://localhost:3000",  # Replace with the origin of your frontend application
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods if needed
    allow_headers=["*"],  # You can specify specific headers if needed
)

API_KEY = "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"

HUGGING_FACE_API_KEY = "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"
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGING_FACE_HUB_API_KEY"] = HUGGING_FACE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY
loader = TextLoader("./data.txt")
# loader = DirectoryLoader(".", glob="*.txt")
# index = VectorstoreIndexCreator(
#     embedding=HuggingFaceEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
# ).from_loaders([loader])

repo_id = "google/flan-t5-base"  # do not provide good conversational support
model_id = "gpt2"
# repo_id = 'lmsys/fastchat-t5-3b-v1.0'
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.environ["HUGGING_FACE_HUB_API_KEY"],
    repo_id=model_id,
    model_kwargs={"temperature": 1e-10, "max_length": 32},
)

# conversation = ConversationChain(llm=llm)
from langchain.chains.conversation.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='history')
conversation_buf = ConversationChain(llm=llm, memory=memory)
memory.load_memory_variables({})


API_URL = "https://api-inference.huggingface.co/models/gpt2-xl"
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})
# la = "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"
@app.get("/")
def read_root(arg: str = Query(..., alias="arg")):
    # response = conversation_buf.predict(input=arg)
    response = query(arg)
    # print(index.query(arg))
    return JSONResponse({"text": response, "sender": "bot"})
