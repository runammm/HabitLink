from .audio_engine import AudioEngine
from .diarizer import SpeakerDiarizer
from .word_analyzer import WordAnalyzer
from .speech_rate_analyzer import SpeechRateAnalyzer
from .utils import load_profanity_list  # Import from utils
from typing import List, Dict, Any

class AudioInterface:
    def __init__(self, audio_engine: AudioEngine, diarizer: SpeakerDiarizer, 
                 word_analyzer: WordAnalyzer, speech_rate_analyzer: SpeechRateAnalyzer):
        self.audio_engine = audio_engine
        self.diarizer = diarizer
        self.word_analyzer = word_analyzer
        self.speech_rate_analyzer = speech_rate_analyzer
        # Load the profanity list once during initialization
        self.profanity_list = load_profanity_list()

    def enroll_user(self, duration: float = 15.0):
        """
        Guides the user through the voice enrollment process.
        """
        enroll_text = "안녕하세요, HabitLink입니다. 지금부터 음성 등록을 시작하겠습니다. 아래의 문장을 평소처럼 편하게 읽어주세요."
        reading_text = "죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다. 오늘 밤에도 별이 바람에 스치운다."
        print("\n--- 🗣️ User Voice Enrollment ---")
        print(f"\"{enroll_text}\"")
        print(f"\"{reading_text}\"")
        input("준비가 되셨으면 Enter 키를 누르고, 위 문장을 읽기 시작하세요...")
        
        # Record the user's voice for enrollment
        enroll_audio_path = self.audio_engine.record(duration, "enrollment_voice.wav")
        print(f"Audio recorded for enrollment and saved to '{enroll_audio_path}'.")
        
        # Create and store the voice embedding
        self.diarizer.enroll_user_voice(enroll_audio_path)

    def record(self, duration: float, output_path: str = "temp.wav"):
        return self.audio_engine.record(duration, output_path)
    
    def record_and_process(self, duration: float, custom_keywords: List[str]) -> Dict[str, Any]:
        """
        Records audio and runs it through the full analysis pipeline, including profanity check.
        """
        # Step 1: Record and Diarize
        input_path = self.record(duration)
        diarized_transcript = self.diarizer.process(input_path)
        
        # If the result is empty, it means no speech was detected.
        if not diarized_transcript:
            print("\n⚠️ 입력된 음성이 없었습니다. (No speech detected.)")
            return {
                "full_transcript": [],
                "detected_custom_keywords": [],
                "detected_profanity": [],
                "speech_rate_analysis": []
            }

        # Step 2: Run parallel analyses
        detected_custom_words = self.word_analyzer.analyze(diarized_transcript, custom_keywords)
        detected_profanity = self.word_analyzer.analyze(diarized_transcript, self.profanity_list)
        speech_rate = self.speech_rate_analyzer.analyze(diarized_transcript)

        # Step 3: Consolidate all results
        return {
            "full_transcript": diarized_transcript,
            "detected_custom_keywords": detected_custom_words,
            "detected_profanity": detected_profanity,
            "speech_rate_analysis": speech_rate
        }
