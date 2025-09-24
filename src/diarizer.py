import os
import whisperx
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Model, Inference
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

load_dotenv()

class SpeakerDiarizer:
    """
    A class to perform transcription and multi-speaker diarization,
    identifying a pre-enrolled user among the speakers.
    """
    def __init__(self, device: str = "cpu", batch_size: int = 16, compute_type: str = "float32"):
        """
        Initializes the Diarizer and loads the required models.
        """
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.asr_model = whisperx.load_model("small", self.device, compute_type=self.compute_type, language="ko")
        
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.")
        
        # Re-introduce the full diarization pipeline
        self.diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        self.diarize_model.to(torch.device(self.device))
        
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
        Transcribes an audio file, diarizes it into multiple speakers,
        and identifies the user among them.
        """
        # 1. Load Audio
        audio = whisperx.load_audio(audio_path)
        sample_rate = 16000 # Whisper models operate at a 16kHz sample rate.

        # 2. Transcribe
        result = self.asr_model.transcribe(audio, batch_size=self.batch_size)
        if not result.get("segments"):
            return []

        # 3. Perform anonymous multi-speaker diarization
        diarization_result = self.diarize_model(audio_path)

        # 4. Identify the user among the anonymous speakers by comparing voice embeddings
        identified_user_label = None
        if self.user_embedding is not None:
            speaker_embeddings = {}
            MIN_SEGMENT_DURATION_S = 1.5

            for speaker_label in diarization_result.labels():
                timeline = diarization_result.label_timeline(speaker_label)
                long_enough_segment = next((s for s in timeline if s.duration > MIN_SEGMENT_DURATION_S), None)
                
                if long_enough_segment is None:
                    print(f"⚠️ No segment long enough for speaker {speaker_label} to create a reliable embedding. Skipping.")
                    continue

                try:
                    embedding = self.embedding_inference.crop(audio_path, long_enough_segment)
                    speaker_embeddings[speaker_label] = np.squeeze(embedding)
                except Exception as e:
                    print(f"Could not create embedding for a segment of speaker {speaker_label}: {e}")
                    continue
            
            if speaker_embeddings:
                labels = list(speaker_embeddings.keys())
                embeds = np.array(list(speaker_embeddings.values()))
                distances = cdist(self.user_embedding.reshape(1, -1), embeds, metric='cosine')
                best_match_idx = np.argmin(distances)

                if distances[0, best_match_idx] < 0.4: # Similarity threshold
                    identified_user_label = labels[best_match_idx]
                    print(f"✅ User identified as anonymous label: {identified_user_label}")

        # 5. Create the final speaker mapping
        diarize_df = pd.DataFrame(diarization_result.itertracks(yield_label=True), columns=['segment', 'track', 'speaker'])
        
        other_speaker_count = 1
        final_speaker_mapping = {}
        for original_label in sorted(diarize_df['speaker'].unique()):
            if original_label == identified_user_label:
                final_speaker_mapping[original_label] = 'USER'
            else:
                final_speaker_mapping[original_label] = f'SPEAKER_{other_speaker_count:02d}'
                other_speaker_count += 1
        
        diarize_df['speaker'] = diarize_df['speaker'].map(final_speaker_mapping)
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        # 6. Align transcription with the final diarization result
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)

        return final_result.get("segments", [])


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
