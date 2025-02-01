from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

from secret_manager import SecretManager
from chatbot import ChatBot

import pydantic
import asyncio
import os

app = FastAPI()
secrets = SecretManager()
secrets.init_secret("OpenAI")
secrets.init_secret("LangChain")
chatbot = ChatBot(secrets.get_secret("OpenAI"))

# Export to environment variables
os.environ["OPENAI_API_KEY"] = secrets.get_secret("OpenAI")
os.environ["LANGSMITH_API_KEY"] = secrets.get_secret("LangChain")

# Mount the static directory to serve index.html
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Data model for POST payload
class AskPayload(pydantic.BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
async def chat(payload: AskPayload):
    response = await asyncio.create_task(chatbot.ask(payload.message))
    return response["answer"]

@app.post("/stream")
async def stream(payload: AskPayload):
    _ = asyncio.create_task(chatbot.ask(payload.message))
    return StreamingResponse(chatbot.response(), media_type="text/plain")
