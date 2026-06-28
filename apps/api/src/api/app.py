from fastapi import FastAPI, Request
from pydantic import BaseModel  
# pydantic is how to define sctrucure and schema for fastapi
from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_llm(provider, model_name, messages, max_tokens=500):
    provider = provider.strip()
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        
    if provider == "Google":
        contents = [message["content"] for message in messages]
        print(f"[Google] Sending contents: {contents}", flush=True)
        return client.models.generate_content(
            model=model_name,
            contents=contents
        ).text
    elif provider == "Groq":
        print(f"[Groq] Sending messages: {messages}", flush=True)
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
        ).choices[0].message.content
    else:
        print(f"[OpenAI] Sending messages: {messages}", flush=True)
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content
        
class ChatRequest(BaseModel):
    provider: str #llm
    model_name: str #llm model name
    messages: list[dict] #llm messages
    max_tokens: int = 500 #llm max tokens defaulted to 500

class ChatResponse(BaseModel):
    message: str #llm response answer

app = FastAPI()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest): ##function handler for /chat decorator
    response_message = run_llm(
        provider=request.provider,
        model_name=request.model_name,
        messages=request.messages
    ) #test
    return ChatResponse(message=response_message) # returns a pyhdantic model with a message in it