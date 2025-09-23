from .audio_engine import AudioEngine
from .stt import STT


class AudioInterface:
    def __init__(self, audio_engine: AudioEngine, stt: STT, language: str = "ko"):
        self.audio_engine = audio_engine
        self.stt = stt
        self.language = language

    def record(self, duration: float):
        return self.audio_engine.record(duration)
    
    def transcribe(self, input_path: str):
        return self.stt.transcribe(input_path, self.language)

    def record_and_transcribe(self, duration: float):
        input_path = self.record(duration)
        return self.transcribe(input_path)


if __name__ == "__main__":
    from .stt import STT_Whisper
    audio_engine = AudioEngine()
    stt = STT_Whisper()
    audio_interface = AudioInterface(audio_engine, stt)
    print(audio_interface.record_and_transcribe(5.0))
