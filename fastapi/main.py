from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from secret_manager import SecretManager
from chatbot import ChatBot

import os

app = FastAPI()
secrets = SecretManager()
secrets.init_secret("OpenAI")
secrets.init_secret("LangChain")

# Export to environment variables
os.environ["OPENAI_API_KEY"] = secrets.get_secret("OpenAI")
os.environ["LANGSMITH_API_KEY"] = secrets.get_secret("LangChain")

# Mount the static directory to serve index.html
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Data model for POST payload
class AskPayload(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask")
async def handle_ask(payload: AskPayload):
    chatbot = ChatBot(secrets.get_secret("OpenAI"))
    response = chatbot.ask(payload.message)

    return {"response": response}

