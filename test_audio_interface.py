import os
import logging
from src.audio_engine import AudioEngine
from src.diarizer import SpeakerDiarizer
from src.word_analyzer import WordAnalyzer
from src.audio_interface import AudioInterface
from dotenv import load_dotenv
import traceback
from collections import defaultdict

# --- Suppress library warnings ---
# Suppress warnings from specific libraries to clean up the output.
logging.getLogger('pyannote').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
# PyTorch Lightning logs a welcome message by default, which we can disable.
os.environ["PYTORCH_SUPPRESS_DEPRECATION_WARNINGS"] = "1"
os.environ["PL_SUPPRESS_FORK"] = "1"
# ---------------------------------

# Load environment variables from .env file
load_dotenv()

print("Initializing components for testing...")

# Check for Hugging Face token
if not os.getenv("HUGGING_FACE_TOKEN"):
    print("Error: HUGGING_FACE_TOKEN environment variable not set.")
    print("Please create a .env file in the root directory and add your token:")
    print('HUGGING_FACE_TOKEN="your_hf_token_here"')
else:
    try:
        # 1. Initialize the core components
        audio_engine = AudioEngine()
        diarizer = SpeakerDiarizer()
        word_analyzer = WordAnalyzer()

        # 2. Initialize the interface with the new components
        audio_interface = AudioInterface(audio_engine, diarizer, word_analyzer)

        # 3. Guide user through voice enrollment
        audio_interface.enroll_user(duration=15.0)

        # 4. Get keywords from user
        print("\n--- 🔍 Keyword Detection Setup ---")
        keyword_input = input("분석할 키워드를 쉼표(,)로 구분하여 입력하세요 (예: 그냥, 근데, 약간): ")
        keywords_to_find = [keyword.strip() for keyword in keyword_input.split(',')]

        # 5. Record a conversation for processing
        input("\n--- 🎙️ Conversation Recording ---\n사용자 등록이 완료되었습니다. 이제 대화 녹음을 시작합니다.\n준비가 되셨으면 Enter 키를 눌러주세요...")
        record_duration = 20.0
        print(f"Recording conversation for {record_duration} seconds...")
        
        # 6. Run the recording, diarization, and analysis process
        analysis_result = audio_interface.record_and_process(record_duration, keywords_to_find)

        # 7. Print the results
        diarized_transcript = analysis_result["full_transcript"]
        found_keywords = analysis_result["detected_keywords"]

        print("\n--- ✅ Diarization Test Result ---")
        if diarized_transcript:
            for segment in diarized_transcript:
                speaker = segment.get('speaker', 'UNKNOWN_SPEAKER')
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                
                print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
        
        print("\n--- 📝 Keyword Analysis ---")
        if not found_keywords:
            print("지정한 키워드가 대화에서 발견되지 않았습니다.")
        else:
            # Group keywords by the detected word for a clean summary
            keyword_summary = defaultdict(list)
            for item in found_keywords:
                keyword_summary[item['keyword'].lower()].append(item)
            
            for keyword, items in keyword_summary.items():
                print(f"'{keyword}': {len(items)}회 검출")
                for item in items:
                    timestamp = item['timestamp']
                    speaker = item['speaker']
                    print(f"  - {timestamp:.2f}s ({speaker})")
        
        print("\n----------------------------")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
