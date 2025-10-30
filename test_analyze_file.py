import os
import logging
import traceback
import asyncio
from collections import defaultdict
from dotenv import load_dotenv
import concurrent.futures

from src.stt import WhisperXDiarizer, GoogleSTTDiarizer
from src.word_analyzer import WordAnalyzer
from src.speech_rate_analyzer import SpeechRateAnalyzer
from src.text_analyzer import TextAnalyzer
from src.stutter_analyzer import StutterAnalyzer
from src.utils import load_profanity_list

# --- Suppress library warnings ---
logging.getLogger('pyannote').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
os.environ["PYTORCH_SUPPRESS_DEPRECATION_WARNINGS"] = "1"
os.environ["PL_SUPPRESS_FORK"] = "1"

# --- Configuration ---
USE_GCP_STT = True  # Set to False to use the local WhisperX model
# ---------------------

# ---------------------------------

async def analyze_audio_file(file_path: str, target_keywords: list = None, enable_stutter_analysis: bool = False):
    """
    Analyzes a given audio file using the full analysis pipeline without user enrollment.
    
    Args:
        file_path (str): Path to the audio file to analyze.
        target_keywords (list): List of specific keywords to detect in the audio.
        enable_stutter_analysis (bool): Whether to run stutter analysis.
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
    if target_keywords:
        print(f"🔍 Will search for keywords: {', '.join(target_keywords)}")

    try:
        # 1. Initialize all analysis components
        if USE_GCP_STT:
            # Ensure you have run 'gcloud auth application-default login'
            # or set GOOGLE_APPLICATION_CREDENTIALS in your .env file.
            print("Using Google Cloud STT for diarization.")
            diarizer = GoogleSTTDiarizer()
        else:
            print("Using local WhisperX for diarization.")
            diarizer = WhisperXDiarizer()
            
        word_analyzer = WordAnalyzer()
        speech_rate_analyzer = SpeechRateAnalyzer()
        text_analyzer = TextAnalyzer()
        stutter_analyzer = StutterAnalyzer() if enable_stutter_analysis else None
        
        # 2. Process the audio file to get a diarized transcript
        # This part is CPU-bound and remains synchronous.
        print("\n--- 🔊 Performing speaker diarization and transcription ---")
        diarized_transcript = diarizer.process(file_path)
        
        if not diarized_transcript:
            print("\n⚠️ No speech detected in the audio file.")
            return
        
        print("✅ Diarization complete.")

        # 3. Run all subsequent analyses on the transcript concurrently
        print("\n--- 📊 Running text analysis modules ---")
        
        # Local, CPU-bound analyses can run concurrently in a thread pool
        # to not block the async event loop.
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            profanity_list = await loop.run_in_executor(pool, load_profanity_list)
            
            profanity_task = loop.run_in_executor(
                pool, word_analyzer.analyze, diarized_transcript, profanity_list
            )
            
            # Add keyword detection task if keywords are provided
            if target_keywords:
                keyword_task = loop.run_in_executor(
                    pool, word_analyzer.analyze, diarized_transcript, target_keywords
                )
            else:
                keyword_task = asyncio.sleep(0, result=[])  # Empty result if no keywords
                
            speech_rate_task = loop.run_in_executor(
                pool, speech_rate_analyzer.analyze, diarized_transcript
            )

            # LLM-based analysis is I/O-bound and runs asynchronously.
            llm_analysis_task = text_analyzer.analyze(diarized_transcript)

            # Stutter analysis (if enabled) - CPU-bound with audio processing
            if enable_stutter_analysis and stutter_analyzer:
                stutter_task = loop.run_in_executor(
                    pool, stutter_analyzer.analyze, file_path, diarized_transcript
                )
                # Gather all results including stutter
                detected_profanity, detected_keywords, speech_rate_analysis, llm_analysis_results, stutter_results = await asyncio.gather(
                    profanity_task, keyword_task, speech_rate_task, llm_analysis_task, stutter_task
                )
            else:
                # Gather all results without stutter
                detected_profanity, detected_keywords, speech_rate_analysis, llm_analysis_results = await asyncio.gather(
                    profanity_task, keyword_task, speech_rate_task, llm_analysis_task
                )
                stutter_results = None

        grammar_analysis = llm_analysis_results.get("grammar_errors", [])
        context_analysis_report = llm_analysis_results.get("context_errors", [])

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

        # Keyword Detection
        if target_keywords:
            print("\n--- 🔍 Keyword Detection ---")
            if not detected_keywords:
                print("No target keywords detected.")
            else:
                summary = defaultdict(list)
                for item in detected_keywords:
                    summary[item['keyword'].lower()].append(item)
                for keyword, items in summary.items():
                    print(f"'{keyword}': {len(items)}회 검출")
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
        
        # Stutter Analysis Report
        if enable_stutter_analysis and stutter_results:
            print("\n--- 🗣️ Stutter Analysis Report ---")
            stats = stutter_results.get("statistics", {})
            repetitions = stutter_results.get("repetitions", [])
            prolongations = stutter_results.get("prolongations", [])
            blocks = stutter_results.get("blocks", [])
            
            print(f"Fluency Score: {stats.get('fluency_percentage', 0):.1f}%")
            print(f"Total Events: {stats.get('total_events', 0)}")
            print(f"\nBreakdown:")
            print(f"  - Repetitions: {len(repetitions)}")
            print(f"  - Prolongations: {len(prolongations)}")
            print(f"  - Blocks: {len(blocks)}")
            
            if repetitions:
                print(f"\n  Repetition Examples:")
                for rep in repetitions[:3]:
                    print(f"    - [{rep.get('timestamp', 0):.1f}s] {rep.get('speaker')}: '{rep.get('full_match')}'")
            
            if prolongations:
                print(f"\n  Prolongation Examples:")
                for prol in prolongations[:3]:
                    print(f"    - [{prol.get('timestamp', 0):.1f}s] {prol.get('speaker')}: '{prol.get('word')}' ({prol.get('duration')}s)")
            
            if blocks:
                print(f"\n  Block Examples:")
                for block in blocks[:3]:
                    print(f"    - [{block.get('timestamp', 0):.1f}s] {block.get('speaker')}: {block.get('duration')}s silence")
        
        print("\n-----------------------------------------")

    except Exception as e:
        print(f"\n--- ❌ An unexpected error occurred ---")
        traceback.print_exc()
        print("-----------------------------------------")

if __name__ == "__main__":
    # audio_file_to_analyze = "data/test_audio/1.wav"
    audio_file_to_analyze = "data/dialect_sample/raw_data/gyeongsang/solo_question/4.wav"
    # audio_file_to_analyze = "data/dialect_sample/raw_data/gangwon/conversation/16.wav"
    # audio_file_to_analyze = "data/broadcast_sample/raw_data/034/broadcast_00033001.wav"
    
    # Get target keywords from user input
    print("=" * 50)
    print("🎯 Audio File Analysis")
    print("=" * 50)
    keywords_input = input("\n검출할 특정 단어를 입력하세요 (콤마로 구분, 예: 지금, 이제, 근데, 약간): ").strip()
    
    target_keywords = None
    if keywords_input:
        target_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
    
    # Ask if stutter analysis should be enabled
    stutter_input = input("\n말더듬 분석을 활성화하시겠습니까? (y/n): ").strip().lower()
    enable_stutter = stutter_input in ['y', 'yes', '예']
    
    print("\n")
    asyncio.run(analyze_audio_file(audio_file_to_analyze, target_keywords, enable_stutter))
