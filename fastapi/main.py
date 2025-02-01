from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket


from secret_manager import SecretManager
from chatbot import ChatBot

import pydantic
import asyncio
import logging
import base64
import json
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

logger = logging.getLogger('uvicorn.error')


# Data model for POST payload
class AskPayload(pydantic.BaseModel):
    message: str

def twiml(resp):
    resp = HTMLResponse(str(resp))
    resp.headers['Content-Type'] = 'text/xml'
    return resp

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(payload: AskPayload):
    response = await asyncio.create_task(chatbot.ask(payload.message))
    return response["answer"]

@app.post("/stream")
async def stream(payload: AskPayload):
    _ = asyncio.create_task(chatbot.ask(payload.message))
    return StreamingResponse(chatbot.response(), media_type="text/plain")

@app.websocket("/process-text")
async def process_text(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        logger.info(f"Received data: {data}")
