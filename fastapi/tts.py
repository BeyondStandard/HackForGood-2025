from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os

from secret_manager import SecretManager
manager = SecretManager()
manager.init_secret("ElevenLabs")

client = ElevenLabs(api_key=manager.get_secret("ElevenLabs"))
audio = client.text_to_speech.convert(
    text="The available night shelter is the Winter Cold Regulation in The Hague, located at Haardstede 1, 2543 VS Den Haag. It is open from 31st January, every day from 18.00 until 10.00. Unfortunately, specific contact data is not provided.",
    voice_id="XfNU2rGpBa01ckF309OY",
    model_id=os.getenv("TTS_MODEL", "eleven_flash_v2_5"),
    output_format="mp3_44100_128",
)
play(audio)