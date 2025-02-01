from twilio.twiml.voice_response import VoiceResponse
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form, Response

from secret_manager import SecretManager
from chatbot import ChatBot

import pydantic
import asyncio
import logging
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

# noinspection PyPep8Naming
@app.post("/process-speech")
async def process_speech(SpeechResult: str = Form(...)):
    logger.info("Received a request to process speech")
    logger.info(f"Speech result: {SpeechResult}")

    gpt_response = await asyncio.create_task(chatbot.ask(SpeechResult))
    response = VoiceResponse()
    response.say(gpt_response["answer"])

    return Response(content=str(response), media_type="application/xml")
