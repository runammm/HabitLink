import os
import pyaudio
import time
from queue import Queue
from typing import Callable, Optional
from google.cloud import speech
import threading


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
    
    def __init__(self, callback: Optional[Callable] = None, audio_callback: Optional[Callable] = None):
        """
        Initialize the streaming STT client.
        
        Args:
            callback: Function to call with (transcript, is_final, speaker, timing_info) on each result
            audio_callback: Function to call with raw audio chunks for visualization
        """
        self.client = speech.SpeechClient()
        self.callback = callback
        self.audio_callback = audio_callback
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = Queue()
        
        # Control flags
        self.is_streaming = False
        self.stop_flag = False
        self.restart_counter = 0
        self.last_transcript_was_final = False
        self.stream_start_time = None
        
        # Microphone reading thread
        self.mic_thread = None
    
    def _get_config(self):
        """Get the STT configuration."""
        return speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.RATE,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,  # Enable word-level timestamps
            model="latest_long",
            use_enhanced=True,
        )
    
    def _get_streaming_config(self):
        """Get the streaming configuration."""
        return speech.StreamingRecognitionConfig(
            config=self._get_config(),
            interim_results=True,
        )
    
    def _fill_buffer(self):
        """Fill buffer with audio chunks."""
        while not self.stop_flag:
            try:
                # Read audio from microphone
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
                
                # Send to audio callback if provided
                if self.audio_callback:
                    try:
                        self.audio_callback(data)
                    except Exception:
                        pass  # Silently ignore callback errors
            except Exception:
                break
    
    def audio_generator(self):
        """Generator that yields audio chunks from the queue."""
        while not self.stop_flag:
            try:
                # Get audio from queue with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                yield chunk
            except:
                continue
    
    def listen_print_loop(self, responses):
        """
        Iterates through server responses and prints/processes them.
        
        Args:
            responses: Streaming responses from the STT API
        """
        for response in responses:
            if self.stop_flag:
                break
            
            if not response.results:
                continue
            
            # The first result in the list is the most recent
            result = response.results[0]
            
            if not result.alternatives:
                continue
            
            # Get the top alternative
            alternative = result.alternatives[0]
            transcript = alternative.transcript
            
            # Extract timing information
            timing_info = None
            if result.result_end_time:
                try:
                    # Handle different timestamp types
                    if hasattr(result.result_end_time, 'total_seconds'):
                        # datetime.timedelta
                        end_time_seconds = result.result_end_time.total_seconds()
                    elif hasattr(result.result_end_time, 'seconds'):
                        # google.protobuf.Duration
                        end_time_seconds = result.result_end_time.seconds + result.result_end_time.nanos / 1e9
                    else:
                        # float or int
                        end_time_seconds = float(result.result_end_time)
                    
                    # Extract word-level timestamps if available
                    word_timestamps = []
                    start_time_seconds = end_time_seconds  # Default to end time
                    
                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            try:
                                # Extract start and end times
                                if hasattr(word_info.start_time, 'total_seconds'):
                                    word_start = word_info.start_time.total_seconds()
                                    word_end = word_info.end_time.total_seconds()
                                elif hasattr(word_info.start_time, 'seconds'):
                                    word_start = word_info.start_time.seconds + word_info.start_time.nanos / 1e9
                                    word_end = word_info.end_time.seconds + word_info.end_time.nanos / 1e9
                                else:
                                    word_start = float(word_info.start_time)
                                    word_end = float(word_info.end_time)
                                
                                word_timestamps.append({
                                    "word": word_info.word,
                                    "start": word_start,
                                    "end": word_end
                                })
                                
                                # Update start time to the first word's start
                                if not word_timestamps or word_start < start_time_seconds:
                                    start_time_seconds = word_start
                            except Exception:
                                continue
                    
                    timing_info = {
                        "start_time": start_time_seconds,
                        "end_time": end_time_seconds,
                        "word_timestamps": word_timestamps
                    }
                except Exception:
                    pass
            
            # Print interim or final results
            if result.is_final:
                print(f"✅ Final: {transcript}")
                if self.callback:
                    self.callback(transcript, True, "SPEAKER_00", timing_info)
                self.last_transcript_was_final = True
            else:
                print(f"⏳ Interim: {transcript}", end='\r', flush=True)
                if self.callback:
                    self.callback(transcript, False, "SPEAKER_00", timing_info)
    
    def start_streaming(self):
        """Start the streaming recognition."""
        self.stop_flag = False
        self.is_streaming = True
        self.stream_start_time = time.time()
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            
            # Start microphone reading thread BEFORE making streaming request
            self.mic_thread = threading.Thread(target=self._fill_buffer, daemon=True)
            self.mic_thread.start()
            
            # Wait a bit to fill buffer
            print("버퍼링 중...")
            buffer_wait = 0
            max_wait = 2  # Maximum wait time in seconds
            while self.audio_queue.qsize() < 10 and buffer_wait < max_wait:
                time.sleep(0.1)
                buffer_wait += 0.1
            
            print(f"버퍼 준비 완료 ({self.audio_queue.qsize()} chunks)")
            
            while not self.stop_flag:
                print(f"\n📡 Opening new stream (restart #{self.restart_counter})...")
                
                # Create streaming config
                streaming_config = self._get_streaming_config()
                
                # Create audio request generator
                audio_generator = self.audio_generator()
                requests = (speech.StreamingRecognizeRequest(audio_content=content)
                           for content in audio_generator)
                
                # Start streaming
                print("🎤 Streaming started...")
                responses = self.client.streaming_recognize(streaming_config, requests)
                
                # Process responses
                try:
                    self.listen_print_loop(responses)
                except Exception as e:
                    if self.stop_flag:
                        break
                    print(f"\n⚠️ Stream error: {e}")
                    print("🔄 Reconnecting...")
                    self.restart_counter += 1
                    time.sleep(0.5)
                    continue
                
                # Check if we need to restart due to time limit
                elapsed_time = time.time() - self.stream_start_time
                if elapsed_time >= self.STREAMING_LIMIT and not self.stop_flag:
                    print(f"\n⏰ Stream limit reached ({self.STREAMING_LIMIT}s). Restarting...")
                    self.stream_start_time = time.time()
                    self.restart_counter += 1
                else:
                    break
        
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        finally:
            self.is_streaming = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            print("🎤 Streaming stopped.")
    
    def stop_streaming(self):
        """Stop the streaming recognition."""
        self.stop_flag = True
        self.is_streaming = False
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.stop_streaming()
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except:
            pass
