import os

from groq import Groq
from dotenv import load_dotenv
import asyncio
from audio import AudioInterface

load_dotenv('.env', override=True)

async def main():
    ai = AudioInterface(samplerate=16000, channels=1)
    await ai.record_to_file(2.5, "voice.wav")

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    with open('voice.wav', 'rb') as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
        )
        print(transcription.text)

for i in range(10):
    asyncio.run(main())
