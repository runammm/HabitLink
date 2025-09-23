import os
from src.audio_engine import AudioEngine
from src.diarizer import SpeakerDiarizer
from src.audio_interface import AudioInterface
from dotenv import load_dotenv
import traceback # Add traceback for detailed error logging

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

        # 3. Define recording duration
        record_duration = 10.0  # seconds

        print(f"Recording audio for {record_duration} seconds...")
        
        # 4. Run the recording and diarization process
        diarized_result = audio_interface.record_and_process(record_duration)

        # 5. Print the results
        print("\n--- Diarization Test Result ---")
        if diarized_result:
            for segment in diarized_result:
                speaker = segment.get('speaker', 'UNKNOWN_SPEAKER')
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                
                print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
        else:
            print("No speech detected or an error occurred during processing.")
        
        print("\n----------------------------")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
