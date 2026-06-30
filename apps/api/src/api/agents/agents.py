from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

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