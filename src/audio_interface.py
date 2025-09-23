from .audio_engine import AudioEngine
from .diarizer import SpeakerDiarizer
from typing import List, Dict, Any

class AudioInterface:
    def __init__(self, audio_engine: AudioEngine, diarizer: SpeakerDiarizer, language: str = "ko"):
        self.audio_engine = audio_engine
        self.diarizer = diarizer
        self.language = language # Language might be handled by the diarizer itself

    def record(self, duration: float):
        return self.audio_engine.record(duration)
    
    def record_and_process(self, duration: float) -> list[dict]:
        """
        Records audio for a given duration and returns the diarized transcript.
        """
        input_path = self.record(duration)
        diarized_transcript = self.diarizer.process(input_path)
        
        # Future step: Add logic here to filter for the primary user's speech
        # based on voice registration, as mentioned in the proposal.
        # For now, we can return the full diarized transcript.

        return diarized_transcript


if __name__ == "__main__":
    # The original __main__ block is kept for reference, but the new test
    # should be run from test_audio_interface.py or a dedicated test script.
    pass
    # from .stt import STT_Whisper
    # audio_engine = AudioEngine()
    # stt = STT_Whisper()
    # audio_interface = AudioInterface(audio_engine, stt)
    # print(audio_interface.record_and_transcribe(5.0))