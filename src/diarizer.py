import os
import whisperx
import torch
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import numpy as np

load_dotenv()

class SpeakerDiarizer:
    """
    A class to perform transcription and identify a pre-enrolled user's speech segments.
    """
    def __init__(self, device: str = "cpu", batch_size: int = 16, compute_type: str = "float32"):
        """
        Initializes the Diarizer and loads the required models.
        """
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.asr_model = whisperx.load_model("base", self.device, compute_type=self.compute_type, language="ko")
        
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.")
        
        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        self.embedding_inference = Inference(self.embedding_model, window="whole")
        self.embedding_inference.to(torch.device(self.device))

        self.user_embedding = None

    def enroll_user_voice(self, audio_path: str):
        """
        Creates and stores a speaker embedding for the user from an audio file.
        """
        try:
            embedding = self.embedding_inference(audio_path)
            self.user_embedding = np.squeeze(embedding)
            print("✅ User voice enrolled successfully.")
        except Exception as e:
            print(f"❌ Failed to enroll user voice: {e}")
            self.user_embedding = None

    def process(self, audio_path: str) -> list[dict]:
        """
        Transcribes an audio file and assigns 'USER' or 'OTHER' labels to speech segments
        based on a pre-enrolled voice embedding.
        """
        # 1. Load Audio
        # whisperx.load_audio automatically resamples the audio to 16000 Hz for the model.
        audio = whisperx.load_audio(audio_path)
        sample_rate = 16000 # Whisper models operate at a 16kHz sample rate.

        # 2. Transcribe to get speech segments
        result = self.asr_model.transcribe(audio, batch_size=self.batch_size)
        segments = result.get("segments", [])

        if not segments:
            return []

        # If no user is enrolled, return segments with an 'UNKNOWN' speaker.
        if self.user_embedding is None:
            print("⚠️ User voice not enrolled. Returning anonymous speaker labels.")
            for seg in segments:
                seg['speaker'] = 'UNKNOWN'
            return segments

        # 3. For each segment, determine if it's the user's voice
        user_embed_np = self.user_embedding.reshape(1, -1)
        
        for segment in segments:
            # Extract audio waveform for the current segment
            start_time = segment['start']
            end_time = segment['end']
            
            # Slice the pre-loaded audio array to get the segment's waveform
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = audio[start_sample:end_sample]

            # Skip very short segments
            if segment_waveform.shape[0] < 1600: # Corresponds to ~0.1s, a safe low-bound
                segment['speaker'] = 'OTHER'
                continue

            # Prepare waveform for the embedding model
            input_waveform = {
                "waveform": torch.from_numpy(segment_waveform).unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            # Create embedding for the segment
            try:
                embedding = self.embedding_inference(input_waveform)
                segment_embedding = np.squeeze(embedding).reshape(1, -1)
                
                # Compare with user's voice
                distance = cdist(user_embed_np, segment_embedding, metric='cosine')[0, 0]
                
                # Assign label based on a distance threshold
                if distance < 0.4:  # This threshold might need tuning
                    segment['speaker'] = 'USER'
                else:
                    segment['speaker'] = 'OTHER'

            except Exception as e:
                print(f"Could not process segment from {start_time:.2f}s to {end_time:.2f}s: {e}")
                segment['speaker'] = 'OTHER'
        
        return segments


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
    DURATION = 5  # seconds
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
