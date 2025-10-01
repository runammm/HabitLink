import os
import logging
import traceback
from collections import defaultdict
from dotenv import load_dotenv
import concurrent.futures

from src.diarizer import SpeakerDiarizer
from src.word_analyzer import WordAnalyzer
from src.speech_rate_analyzer import SpeechRateAnalyzer
from src.grammar_analyzer import GrammarAnalyzer
from src.context_analyzer import ContextAnalyzer

# --- Suppress library warnings ---
logging.getLogger('pyannote').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
os.environ["PYTORCH_SUPPRESS_DEPRECATION_WARNINGS"] = "1"
os.environ["PL_SUPPRESS_FORK"] = "1"
# ---------------------------------

def analyze_audio_file(file_path: str):
    """
    Analyzes a given audio file using the full analysis pipeline without user enrollment.
    """
    load_dotenv()

    # --- Pre-computation Checks ---
    if not os.path.exists(file_path):
        print(f"❌ Error: Audio file not found at '{file_path}'")
        print("Please place an audio file at that location to run the analysis.")
        return

    if not os.getenv("HUGGING_FACE_TOKEN") or not os.getenv("GROQ_API_KEY"):
        print("❌ Error: Required API keys (HUGGING_FACE_TOKEN, GROQ_API_KEY) not found in .env file.")
        return

    print(f"🚀 Initializing components to analyze '{file_path}'...")

    try:
        # 1. Initialize all analysis components
        # Note: We do not initialize AudioEngine as we are not recording.
        diarizer = SpeakerDiarizer()
        word_analyzer = WordAnalyzer()
        speech_rate_analyzer = SpeechRateAnalyzer()
        grammar_analyzer = GrammarAnalyzer()
        # For file analysis, we analyze all speakers, so we pass a wildcard or default speaker ID
        # The logic inside the analyzer will determine what to do based on the prompt.
        context_analyzer = ContextAnalyzer()

        # 2. Process the audio file to get a diarized transcript
        # Since we skipped enroll_user, the diarizer will use anonymous labels (SPEAKER_00, etc.)
        print("\n--- 🔊 Performing speaker diarization and transcription ---")
        diarized_transcript = diarizer.process(file_path)
        
        if not diarized_transcript:
            print("\n⚠️ No speech detected in the audio file.")
            return
        
        print("✅ Diarization complete.")

        # 3. Run all subsequent analyses on the transcript
        print("\n--- 📊 Running text analysis modules ---")
        
        # Word and speech rate analyses are local and fast.
        keywords_to_find = []
        # For this test, we won't analyze for custom keywords.
        found_keywords = word_analyzer.analyze(diarized_transcript, keywords_to_find)
        from src.utils import load_profanity_list
        profanity_list = load_profanity_list()
        detected_profanity = word_analyzer.analyze(diarized_transcript, profanity_list)
        speech_rate_analysis = speech_rate_analyzer.analyze(diarized_transcript)

        # Grammar and Context analysis involve network calls.
        # We run them sequentially as per the original design.
        grammar_analysis = grammar_analyzer.analyze(diarized_transcript)
        context_analysis_report = context_analyzer.analyze(diarized_transcript)

        print("✅ Text analysis complete.")

        # 4. Print all results in a comprehensive report
        print("\n\n--- 📋 COMPREHENSIVE ANALYSIS REPORT ---")

        # Diarization Result
        print("\n--- ✅ Full Transcript ---")
        for segment in diarized_transcript:
            speaker = segment.get('speaker', 'UNKNOWN')
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
            print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")

        # Profanity Detection
        print("\n--- 🤬 Profanity Detection ---")
        if not detected_profanity:
            print("No profanity detected.")
        else:
            summary = defaultdict(list)
            for item in detected_profanity:
                summary[item['keyword'].lower()].append(item)
            for profanity, items in summary.items():
                print(f"'{profanity}': {len(items)}회 검출")
                for item in items:
                    print(f"  - {item['timestamp']:.2f}s ({item['speaker']})")

        # Speech Rate Analysis
        print("\n--- 🏃 Speech Rate Analysis ---")
        if not speech_rate_analysis:
            print("Could not analyze speech rate.")
        else:
            # Similar printing logic as the other test file
            total_word_count = 0
            total_duration = 0
            for segment in speech_rate_analysis:
                speaker = segment.get('speaker', 'UNKNOWN')
                wpm = segment.get('wpm', 0)
                print(f"  - {speaker}: {wpm:.2f} WPM")
                total_word_count += segment.get('word_count', 0)
                total_duration += segment.get('duration', 0)
            if total_duration > 0:
                overall_wpm = (total_word_count / total_duration) * 60
                print(f"\n  Overall Average: {overall_wpm:.2f} WPM")

        # Grammar Analysis
        print("\n--- 🧐 Grammar Analysis ---")
        if not grammar_analysis:
            print("No grammatical errors found.")
        else:
            for error in grammar_analysis:
                details = error.get('error_details', {})
                print(f"\n  - [{error.get('timestamp'):.2f}s, {error.get('speaker')}] '{details.get('original')}' → '{details.get('corrected')}'")
                print(f"    - Reason: {details.get('explanation')}")

        # Contextual Coherence Report
        print("\n--- 🧠 Contextual Coherence Report ---")
        if not context_analysis_report:
            print("No contextual errors found.")
        else:
            print(f"Found {len(context_analysis_report)} contextual error(s):")
            for error in context_analysis_report:
                print(f"\n  - [{error.get('timestamp'):.2f}s, {error.get('speaker')}] \"{error.get('utterance')}\"")
                print(f"    - Analysis: {error.get('reasoning')}")
        
        print("\n-----------------------------------------")

    except Exception as e:
        print(f"\n--- ❌ An unexpected error occurred ---")
        traceback.print_exc()
        print("-----------------------------------------")

if __name__ == "__main__":
    audio_file_to_analyze = "data/test_audio/2.wav"
    
    analyze_audio_file(audio_file_to_analyze)
