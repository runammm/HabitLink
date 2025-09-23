from src import STT_Whisper
from src import AudioEngine
from src import AudioInterface

audio_engine = AudioEngine()
stt = STT_Whisper()
audio_interface = AudioInterface(audio_engine, stt)
print(audio_interface.record_and_transcribe(5.0))
