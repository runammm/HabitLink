import os
import whisperx
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Model, Inference
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

load_dotenv()

class WhisperXDiarizer:
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
        self.asr_model = whisperx.load_model("base", self.device, compute_type=self.compute_type, language="ko")
        
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
        sample_rate = 16000  # Whisper models operate at a 16kHz sample rate.

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
        
        final_speaker_mapping = {}
        # If a user was identified via enrollment, perform binary User/Others classification
        if identified_user_label:
            for original_label in sorted(diarize_df['speaker'].unique()):
                if original_label == identified_user_label:
                    final_speaker_mapping[original_label] = 'User'
                else:
                    final_speaker_mapping[original_label] = 'Others'
        # Otherwise (no enrollment), use anonymous incremental speaker labels
        else:
            for i, original_label in enumerate(sorted(diarize_df['speaker'].unique())):
                final_speaker_mapping[original_label] = f'SPEAKER_{i:02d}'

        diarize_df['speaker'] = diarize_df['speaker'].map(final_speaker_mapping)
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        # 6. Align transcription with the final diarization result
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        # Set return_char_alignments=False for word-level timestamps
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)

        # Return the segments which now include word-level timestamps
        return final_result.get("segments", [])


from google.cloud import speech
import soundfile as sf

class GoogleSTTDiarizer:
    """
    A class to perform transcription and speaker diarization using Google Cloud STT.
    """
    def __init__(self):
        """
        Initializes the GoogleSTTDiarizer.
        """
        self.client = speech.SpeechClient()

    def process(self, audio_path: str) -> list[dict]:
        """
        Transcribes an audio file and performs speaker diarization using Google Cloud STT.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            list[dict]: A list of transcribed segments with speaker labels and timestamps,
                        matching the format of the WhisperX-based SpeakerDiarizer.
        """
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio_info = sf.info(audio_path)
        sample_rate = audio_info.samplerate

        audio = speech.RecognitionAudio(content=content)

        # Configuration for speaker diarization
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=6,  # Adjust as needed
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="ko-KR",
            diarization_config=diarization_config,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        print("Sending request to Google Cloud STT...")
        operation = self.client.long_running_recognize(config=config, audio=audio)
        
        print("Waiting for Google Cloud STT to complete...")
        response = operation.result(timeout=300)  # Timeout in seconds
        print("Received response from Google Cloud STT.")

        if not response.results:
            return []
        
        # Collect all words with speaker tags from all results
        all_words = []
        for result in response.results:
            if result.alternatives and result.alternatives[0].words:
                words_in_result = result.alternatives[0].words
                for word in words_in_result:
                    all_words.append(word)
        
        if not all_words:
            return []
        
        # Remove duplicates based on start_time
        # Google Cloud STT can return duplicate words with different speaker tags in multiple results
        # Keep the LAST occurrence as later results contain more complete diarization info
        seen_timestamps = {}
        for word in all_words:
            # Handle timedelta, Duration objects, and float timestamps
            try:
                timestamp = word.start_time.total_seconds()
            except (AttributeError, TypeError):
                try:
                    timestamp = word.start_time.seconds + word.start_time.nanos / 1e9
                except AttributeError:
                    timestamp = float(word.start_time)
            
            timestamp_key = round(timestamp, 3)
            seen_timestamps[timestamp_key] = word
        
        # Sort by timestamp to maintain chronological order
        all_words = [seen_timestamps[ts] for ts in sorted(seen_timestamps.keys())]
        
        # Helper function to safely extract timestamp
        def get_time(time_obj):
            try:
                # Try timedelta (has total_seconds method)
                return time_obj.total_seconds()
            except (AttributeError, TypeError):
                try:
                    # Try Duration object (Google Cloud STT format)
                    return time_obj.seconds + time_obj.nanos / 1e9
                except AttributeError:
                    # Try float/int
                    return float(time_obj)
        
        # Group words into segments by speaker
        segments = []
        current_speaker_tag = all_words[0].speaker_tag
        current_segment_text = []
        current_segment_words = []
        segment_start_time = get_time(all_words[0].start_time)

        for i, word in enumerate(all_words):
            if word.speaker_tag != current_speaker_tag:
                # Finalize the previous segment
                segments.append({
                    "text": " ".join(current_segment_text).strip(),
                    "start": segment_start_time,
                    "end": get_time(all_words[i - 1].end_time),
                    "speaker": f"SPEAKER_{current_speaker_tag:02d}",
                    "words": current_segment_words
                })
                
                # Start a new segment
                current_speaker_tag = word.speaker_tag
                current_segment_text = [word.word]
                current_segment_words = [{
                    "word": word.word,
                    "start": get_time(word.start_time),
                    "end": get_time(word.end_time)
                }]
                segment_start_time = get_time(word.start_time)
            else:
                current_segment_text.append(word.word)
                current_segment_words.append({
                    "word": word.word,
                    "start": get_time(word.start_time),
                    "end": get_time(word.end_time)
                })
        
        # Add the last segment
        segments.append({
            "text": " ".join(current_segment_text).strip(),
            "start": segment_start_time,
            "end": get_time(all_words[-1].end_time),
            "speaker": f"SPEAKER_{current_speaker_tag:02d}",
            "words": current_segment_words
        })

        return segments


import pyaudio
import time
from queue import Queue
from typing import Callable, Optional

class GoogleSTTStreaming:
    """
    A class to perform real-time streaming transcription using Google Cloud STT with websocket.
    Includes automatic reconnection logic to handle the 5-minute streaming limit.
    """
    
    # Audio recording parameters
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms chunks
    
    # Streaming time limit (4 minutes to be safe, GCP limit is 5 minutes)
    STREAMING_LIMIT = 240
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize the streaming STT client.
        
        Args:
            callback: Optional callback function to receive transcription results
                     Signature: callback(transcript: str, is_final: bool, speaker: str)
        """
        self.client = speech.SpeechClient()
        self.callback = callback
        self.audio_queue = Queue()
        self.is_running = False
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        
    def get_current_time(self):
        """Return current time in milliseconds."""
        return int(round(time.time() * 1000))
    
    def duration_to_secs(self, duration):
        """Convert duration object to seconds."""
        return duration.seconds + duration.nanos / 1e9
    
    def audio_generator(self):
        """
        Generator that yields audio chunks from the queue.
        This is used to stream audio to Google Cloud STT.
        """
        while self.is_running:
            # Use a blocking get() to ensure there's at least one chunk of data
            chunk = self.audio_queue.get()
            if chunk is None:
                return
            
            data = [chunk]
            
            # Now consume whatever other data's still buffered
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                if chunk is None:
                    return
                data.append(chunk)
            
            yield b''.join(data)
    
    def listen_print_loop(self, responses):
        """
        Iterates through server responses and prints them.
        
        The responses passed is a generator that will block until a response
        is provided by the server.
        """
        for response in responses:
            if not response.results:
                continue
            
            # The results list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's is_final, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue
            
            # Extract transcript
            transcript = result.alternatives[0].transcript
            
            # Get timing information
            result_seconds = 0
            result_nanos = 0
            
            if result.result_end_time:
                result_seconds = result.result_end_time.seconds
                result_nanos = result.result_end_time.nanos
            
            stream_time = self.STREAMING_LIMIT * self.restart_counter
            self.result_end_time = int((result_seconds * 1000) + (result_nanos / 1000000))
            
            corrected_time = (
                self.result_end_time - self.bridging_offset + stream_time
            )
            
            # Determine speaker (placeholder since streaming doesn't support diarization well)
            speaker = "SPEAKER_00"
            
            if result.is_final:
                # Final result
                self.is_final_end_time = self.result_end_time
                self.last_transcript_was_final = True
                
                # Call callback if provided
                if self.callback:
                    self.callback(transcript, True, speaker)
                
                # Store the transcript
                print(f"✅ Final: {transcript}")
            else:
                # Interim result
                self.last_transcript_was_final = False
                
                # Call callback if provided
                if self.callback:
                    self.callback(transcript, False, speaker)
                
                # Print interim result (overwrite previous line)
                print(f"⏳ Interim: {transcript}", end='\r')
    
    def start_streaming(self, duration: Optional[float] = None):
        """
        Start streaming audio from microphone to Google Cloud STT.
        
        Args:
            duration: Optional duration in seconds. If None, streams indefinitely.
        """
        self.is_running = True
        
        # Configure audio input
        audio_interface = pyaudio.PyAudio()
        
        try:
            # Open microphone stream
            mic_stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            
            print("🎤 Streaming started...")
            
            start_time = time.time()
            
            while self.is_running:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                self.audio_input = []
                audio_stream = self.audio_generator()
                
                # Configure streaming recognition
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.RATE,
                    language_code="ko-KR",
                    enable_automatic_punctuation=True,
                    max_alternatives=1,
                )
                
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=True,
                )
                
                # Start streaming recognize
                print(f"\n📡 Opening new stream (restart #{self.restart_counter})...")
                
                # Create audio request generator
                audio_requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_stream
                )
                
                # Make streaming request
                responses = self.client.streaming_recognize(
                    config=streaming_config,
                    requests=audio_requests,
                )
                
                # Start microphone reading thread
                mic_thread_active = True
                
                def fill_buffer():
                    """Continuously collect audio from microphone and add to queue."""
                    nonlocal mic_thread_active
                    stream_start_time = self.get_current_time()
                    
                    while mic_thread_active and self.is_running:
                        try:
                            # Check if we've hit the streaming limit
                            if (self.get_current_time() - stream_start_time) > (self.STREAMING_LIMIT * 1000):
                                break
                            
                            # Read audio chunk
                            data = mic_stream.read(self.CHUNK, exception_on_overflow=False)
                            
                            # Add to queue for streaming
                            self.audio_queue.put(data)
                            
                            # Store for bridging
                            self.audio_input.append(data)
                            
                        except Exception as e:
                            print(f"⚠️ Error reading microphone: {e}")
                            break
                
                import threading
                mic_thread = threading.Thread(target=fill_buffer)
                mic_thread.start()
                
                # Process responses
                try:
                    self.listen_print_loop(responses)
                except Exception as e:
                    print(f"\n⚠️ Stream error: {e}")
                
                # Stop microphone thread
                mic_thread_active = False
                mic_thread.join()
                
                # Signal end of stream
                self.audio_queue.put(None)
                
                # Check if we should continue
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Prepare for reconnection
                if self.is_running:
                    if self.last_transcript_was_final:
                        self.bridging_offset = 0
                    else:
                        # Carry over partial audio for continuity
                        self.bridging_offset = self.result_end_time
                    
                    # Increment restart counter
                    self.restart_counter += 1
                    
                    # Brief pause before reconnecting
                    time.sleep(0.1)
            
        finally:
            # Cleanup
            mic_stream.stop_stream()
            mic_stream.close()
            audio_interface.terminate()
            self.is_running = False
            print("\n🎤 Streaming stopped.")
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_running = False
        self.audio_queue.put(None)


if __name__ == '__main__':
    # This is a test script to verify the functionality.
    # To run this, you need a multi-speaker audio file.
    # 1. Place an audio file (e.g., 'multi_speaker.wav') in the root directory.
    # 2. Set your HUGGING_FACE_TOKEN in a .env file.
    # 3. Run this script directly: python -m src.diarizer

    print("Testing WhisperXDiarizer...")
    
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
        diarizer = WhisperXDiarizer()
        
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
