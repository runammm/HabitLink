import os
from src.audio_engine import AudioEngine
from src.diarizer import SpeakerDiarizer
from src.audio_interface import AudioInterface
from dotenv import load_dotenv
import traceback

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
        diarizer = SpeakerDiarizer()  # Uses HF_TOKEN from .env

        # 2. Initialize the interface with the new components
        audio_interface = AudioInterface(audio_engine, diarizer)

        # 3. Guide user through voice enrollment
        audio_interface.enroll_user(duration=15.0)

        # 4. Record a conversation for processing
        input("\n--- 🎙️ Conversation Recording ---\n사용자 등록이 완료되었습니다. 이제 대화 녹음을 시작합니다.\n준비가 되셨으면 Enter 키를 눌러주세요...")
        record_duration = 20.0
        print(f"Recording conversation for {record_duration} seconds...")
        
        # 5. Run the recording and diarization process
        diarized_result = audio_interface.record_and_process(record_duration)

        # 6. Print the results
        print("\n--- ✅ Diarization Test Result ---")
        if diarized_result:
            for segment in diarized_result:
                speaker = segment.get('speaker', 'UNKNOWN_SPEAKER')
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                
                print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
        
        print("\n----------------------------")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
