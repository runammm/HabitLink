from src.audio import AudioInterface
from src.stt import stt
import asyncio
import uuid

audio_interface = AudioInterface(samplerate=16000, channels=1)

async def cycle():
    segment_seconds = 10
    max_concurrency = 2
    pending = set()

    while True:
        filename = f"voice.wav"
        print(f"Recording...")
        out_path = await audio_interface.record_to_file(segment_seconds, filename)
        print(f"Recording done...")
        task = asyncio.create_task(stt(out_path))
        # consume exceptions to avoid "Task exception was never retrieved"
        task.add_done_callback(lambda t: t.exception())
        pending.add(task)

        if len(pending) >= max_concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

asyncio.run(cycle())