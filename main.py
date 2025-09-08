import os
import asyncio
import tempfile
from src.audio import AudioInterface
from src.stt import stt

audio_interface = AudioInterface(samplerate=16000, channels=1)


async def cycle():
    segment_seconds = 10
    max_concurrency = 2
    pending = set()

    while True:
        with tempfile.NamedTemporaryFile(prefix="voice_", suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        print(f"Recording...")
        out_path = await audio_interface.record_to_file(segment_seconds, tmp_path)
        print(f"Recording done... -> {out_path}")
        task = asyncio.create_task(stt(out_path))

        def _cleanup_and_log(t: asyncio.Task, path=out_path):
            try:
                _ = t.result()  # 예외 발생 시 여기서 raise → except에서 로깅
            except Exception as e:
                print(f"[STT ERROR] {e!r}")
            finally:
                try:
                    os.remove(path)
                    # print(f"[CLEANUP] removed {path}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"[CLEANUP ERROR] {e!r}")

        # consume exceptions to avoid "Task exception was never retrieved"
        task.add_done_callback(_cleanup_and_log)
        pending.add(task)

        if len(pending) >= max_concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

asyncio.run(cycle())
