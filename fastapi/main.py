from twilio.twiml.voice_response import VoiceResponse
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware

from secret_manager import SecretManager
from chatbot import ChatBot

import pydantic
import asyncio
import logging
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

ACTIVE_STREAMS = {}

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
    return response

@app.post("/stream")
async def stream(payload: AskPayload):
    stream_generator = chatbot.stream_ask(payload.message)
    return StreamingResponse(stream_generator, media_type="text/plain")

# noinspection PyPep8Naming
@app.post("/process-speech")
async def process_speech(
        CallSid: str = Form(...),
        SpeechResult: str = Form(...),
):
    ACTIVE_STREAMS[CallSid] = chatbot.stream_ask(SpeechResult)

    response = VoiceResponse()
    response.say("Let me check that for you.")
    response.redirect(
        url=f"https://iceboxdev-fastapi--8000.prod1a.defang.dev/output?call_sid={CallSid}",
        method="GET"
    )

    logger.info(f"Processing speech: {SpeechResult}")
    return Response(content=str(response), media_type="application/xml")

@app.get("/output")
async def output(call_sid: str):
    out = ""

    async for chunk in ACTIVE_STREAMS[call_sid]:
        out += chunk
        if "." in chunk or "?" in chunk or "!" in chunk:
            response = VoiceResponse()
            response.say(out)
            response.append(response.redirect(f"https://iceboxdev-fastapi--8000.prod1a.defang.dev/output?call_sid={call_sid}", method='GET'))
            logger.info(f"Returning: {out}")
            return Response(content=str(response), media_type="application/xml")

    else:
        logger.info("Finished outputting stream.")
        del ACTIVE_STREAMS[call_sid]

        response = VoiceResponse()
        return Response(content=str(response), media_type="application/xml")
