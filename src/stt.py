from groq import Groq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv('.env', override=True)

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

async def stt(file_path):
    def _call():
        try:
            with open(file_path, 'rb') as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    language="ko",
                    )
                print(transcription.text)
                return transcription.text
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
    return await asyncio.to_thread(_call)