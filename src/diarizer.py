import os
import whisperx
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline # Updated import for new API
import pandas as pd # Import pandas for data manipulation

load_dotenv()

class SpeakerDiarizer:
    """
    A class to perform speaker diarization and transcription using WhisperX.
    """
    def __init__(self, device: str = "cpu", batch_size: int = 16, compute_type: str = "float32"):
        """
        Initializes the SpeakerDiarizer and loads the required models.

        Args:
            device (str): Device to run the models on ("cuda" or "cpu").
            batch_size (int): Batch size for transcription.
            compute_type (str): The compute type for the model (e.g., "int8", "float16", "float32").
        """
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        # Note: You can specify the whisper model size here, e.g., "large-v2"
        self.asr_model = whisperx.load_model("base", self.device, compute_type=self.compute_type, language="ko")
        
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.")
        
        # Updated diarization model loading for pyannote.audio 3.x
        self.diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        self.diarize_model.to(torch.device(self.device))


    def process(self, audio_path: str, min_speakers: int = None, max_speakers: int = None) -> list[dict]:
        """
        Transcribes and diarizes an audio file.

        Args:
            audio_path (str): The path to the input audio file.
            min_speakers (int, optional): The minimum number of speakers.
            max_speakers (int, optional): The maximum number of speakers.

        Returns:
            list[dict]: A list of transcript segments, each with 'start', 'end', 'text', and 'speaker' keys.
                        Example: [{'start': 0.5, 'end': 2.3, 'text': 'Hello there.', 'speaker': 'SPEAKER_00'}]
        """
        # 1. Load Audio
        audio = whisperx.load_audio(audio_path)

        # 2. Transcribe
        result = self.asr_model.transcribe(audio, batch_size=self.batch_size)

        # 3. Align Whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        # 4. Diarize
        # Updated to use the new pyannote.audio pipeline API
        diarization_result = self.diarize_model(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

        # Convert pyannote annotation to a pandas DataFrame whisperx can use
        diarize_df = pd.DataFrame(diarization_result.itertracks(yield_label=True), columns=['segment', 'track', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        # Assign speaker labels to words
        result = whisperx.assign_word_speakers(diarize_df, result)

        # Format output for clarity
        formatted_segments = []
        for segment in result["segments"]:
            # Ensure all keys are present, providing defaults if they are not.
            formatted_segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
                "speaker": segment.get("speaker")
            })
        
        return formatted_segments

if __name__ == '__main__':
    # This is a test script to verify the functionality.
    # To run this, you need a multi-speaker audio file.
    # 1. Place an audio file (e.g., 'multi_speaker.wav') in the root directory.
    # 2. Set your HUGGING_FACE_TOKEN in a .env file.
    # 3. Run this script directly: python -m src.diarizer

    print("Testing SpeakerDiarizer...")
    
    # Create a dummy audio file for testing if it doesn't exist.
    # This dummy file won't produce a meaningful diarization result but will prevent crashes.
    from scipy.io.wavfile import write
    import numpy as np

    SAMPLE_RATE = 16000
    DURATION = 5 # seconds
    TEST_AUDIO_PATH = "temp_diarizer_test.wav"

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"Creating a dummy audio file at '{TEST_AUDIO_PATH}' for testing.")
        # Create a simple tone
        t = np.linspace(0., DURATION, int(SAMPLE_RATE * DURATION))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * 440. * t)
        write(TEST_AUDIO_PATH, SAMPLE_RATE, data.astype(np.int16))

    try:
        # Initialize the diarizer
        diarizer = SpeakerDiarizer()
        
        # Process the audio file
        print(f"Processing '{TEST_AUDIO_PATH}'...")
        diarized_result = diarizer.process(TEST_AUDIO_PATH)
        
        # Print the result
        print("\\nDiarization Result:")
        for segment in diarized_result:
            speaker = segment['speaker']
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            if speaker:
                print(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the dummy audio file
        if os.path.exists(TEST_AUDIO_PATH):
            os.remove(TEST_AUDIO_PATH)
            print(f"Removed dummy audio file '{TEST_AUDIO_PATH}'.")
