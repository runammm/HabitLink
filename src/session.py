import os
import time
import asyncio
import threading
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional
import concurrent.futures
import numpy as np

from .audio_engine import AudioEngine
from .stt import GoogleSTTStreaming
from .word_analyzer import WordAnalyzer
from .speech_rate_analyzer import SpeechRateAnalyzer
from .text_analyzer import TextAnalyzer
from .stutter_analyzer import StutterAnalyzer
from .stutter_detector import StutterDetector
from .dialect_analyzer import DialectAnalyzer
from .utils import load_profanity_list
from .report_generator import ReportGenerator


class HabitLinkSession:
    """
    Main class for managing a HabitLink analysis session with streaming architecture.
    """
    
    def __init__(self):
        """Initialize the HabitLink session with default settings."""
        self.audio_engine = None
        self.streaming_stt = None
        self.word_analyzer = None
        self.speech_rate_analyzer = None
        self.text_analyzer = None
        self.stutter_analyzer = None
        self.stutter_detector = None  # Real-time audio-based stutter detection
        self.dialect_analyzer = None  # Dialect detection
        self.profanity_list = []
        
        # User configuration
        self.enabled_analyses = {
            "keyword_detection": False,
            "profanity_detection": False,
            "speech_rate": False,
            "grammar": False,
            "context": False,
            "stutter": False,
            "dialect": False
        }
        self.custom_keywords = []
        self.target_wpm = None
        
        # Threading components
        self.feedback_queue = Queue()
        self.ui_feedback_queue = Queue()
        self.audio_queue = Queue()
        self.stop_event = threading.Event()
        
        # Recording state management (for instant detection)
        self.is_recording = False
        self.is_initialized = False
        
        # Streaming buffers
        self.transcript_buffer = []  # Buffer for recent transcripts
        self.audio_buffer = deque(maxlen=16000 * 30)  # 30 seconds of audio at 16kHz
        self.last_analysis_time = time.time()
        
        # Speech rate monitoring (per-sentence analysis)
        self.speech_rate_audio_buffer = deque(maxlen=16000 * 10)  # Keep for potential future use
        
        # Dialect detection monitoring (10-second windows)
        self.dialect_audio_buffer = deque(maxlen=16000 * 10)  # 10 seconds at 16kHz
        self.dialect_text_buffer = []  # Transcripts in current 10s window
        self.last_dialect_check = time.time()
        
        # Store analysis results for summary
        self.all_keyword_detections = []
        self.all_profanity_detections = []
        self.all_speech_rate_results = []
        self.all_grammar_errors = []
        self.all_context_errors = []
        self.stutter_results = None
        self.dialect_results = []  # Store dialect analysis results
        
        # Track processed transcripts to avoid duplicates
        self.processed_transcript_ids = set()
        self.llm_analyzed_transcript_ids = set()  # Track transcripts already analyzed by LLM
        
        # Track word counts in current interim sequence (reset on Final)
        # Format: {(type, keyword_lowercase): count}
        self.interim_word_counts = {}
        
        # Track detected items to prevent duplicates in report
        # Format: {(type, keyword_lowercase, rounded_timestamp)}
        self.detected_items_for_report = set()
        
        self.last_interim_text = ""
        self.last_interim_item = None  # Store last interim to handle session end
        
        # Session metadata
        self.session_start_time = None
        
        # Report generator
        self.report_generator = ReportGenerator()
        
        # Early initialization for instant detection
        print("\n🚀 Pre-initializing components for instant detection...")
        self._early_initialize()
    
    def _early_initialize(self):
        """
        Early initialization of all components for instant detection.
        This runs immediately when HabitLinkSession is created.
        """
        try:
            # Initialize audio engine (used for calibration)
            self.audio_engine = AudioEngine(samplerate=16000, channels=1)
            print("✅ Audio engine initialized")
            
            # Initialize all analyzers upfront
            self.word_analyzer = WordAnalyzer()
            self.speech_rate_analyzer = SpeechRateAnalyzer()
            self.text_analyzer = TextAnalyzer()
            self.stutter_analyzer = StutterAnalyzer()
            self.stutter_detector = StutterDetector()
            
            # Initialize dialect analyzer (optional - only if model exists)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dialect_binary_classifier", "final_model")
            vocabulary_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "dialect_vocabulary.txt")
            self.dialect_analyzer = DialectAnalyzer(model_path, vocabulary_path)
            
            print("✅ Analysis modules initialized")
            
            # Load profanity list
            self.profanity_list = load_profanity_list()
            print(f"✅ Profanity list loaded ({len(self.profanity_list)} words)")
            
            self.is_initialized = True
            print("✅ System ready for instant detection\n")
            return True
            
        except Exception as e:
            print(f"❌ Error during early initialization: {e}")
            self.is_initialized = False
            return False
        
    def initialize_components(self):
        """
        Legacy method for compatibility. Components are now initialized in _early_initialize().
        """
        if not self.is_initialized:
            return self._early_initialize()
        return True
    
    def select_analyses(self):
        """Interactive menu for users to select which analyses to enable."""
        print("\n" + "="*60)
        print("📊 분석 모듈 선택")
        print("="*60)
        print("\n사용할 분석 모듈을 선택하세요:")
        print("1. 특정 반복 단어 검출")
        print("2. 비속어 검출")
        print("3. 발화 속도 분석")
        print("4. 문법 분석")
        print("5. 맥락 분석")
        print("6. 말더듬 분석")
        print("7. 방언 분석")
        print("\n여러 개를 선택하려면 쉼표로 구분하세요 (예: 1,3,4,7)")
        
        selection = input("\n선택: ").strip()
        
        if not selection:
            print("⚠️ 아무것도 선택하지 않았습니다. 모든 분석을 비활성화합니다.")
            return
        
        selected_numbers = [s.strip() for s in selection.split(",")]
        
        if "1" in selected_numbers:
            self.enabled_analyses["keyword_detection"] = True
            print("\n✅ 특정 반복 단어 검출이 활성화되었습니다.")
        
        if "2" in selected_numbers:
            self.enabled_analyses["profanity_detection"] = True
            print("✅ 비속어 검출이 활성화되었습니다.")
        
        if "3" in selected_numbers:
            self.enabled_analyses["speech_rate"] = True
            print("✅ 발화 속도 분석이 활성화되었습니다.")
        
        if "4" in selected_numbers:
            self.enabled_analyses["grammar"] = True
            print("✅ 문법 분석이 활성화되었습니다.")
        
        if "5" in selected_numbers:
            self.enabled_analyses["context"] = True
            print("✅ 맥락 분석이 활성화되었습니다.")
        
        if "6" in selected_numbers:
            self.enabled_analyses["stutter"] = True
            print("✅ 말더듬 분석이 활성화되었습니다.")
        
        if "7" in selected_numbers:
            if self.dialect_analyzer and self.dialect_analyzer.is_available():
                self.enabled_analyses["dialect"] = True
                print("✅ 방언 분석이 활성화되었습니다 (표준어 vs 비표준어 판별).")
            else:
                print("⚠️ 방언 분석 모델이 준비되지 않았습니다.")
                print("   'notebooks/dialect_model_training.ipynb'를 Colab에서 실행하여 모델을 먼저 학습시켜주세요.")
    
    def prepare_session(self):
        """Prepare the session based on selected analyses."""
        print("\n" + "="*60)
        print("🔧 세션 준비 중...")
        print("="*60)
        
        # If keyword detection is enabled, get keywords from user
        if self.enabled_analyses["keyword_detection"]:
            print("\n--- 특정 반복 단어 검출 설정 ---")
            keywords_input = input("검출할 단어를 입력하세요 (콤마로 구분, 예: 지금, 이제, 근데, 약간): ").strip()
            if keywords_input:
                self.custom_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
                print(f"✅ 검출할 단어: {', '.join(self.custom_keywords)}")
            else:
                print("⚠️ 단어가 입력되지 않았습니다. 키워드 검출을 비활성화합니다.")
                self.enabled_analyses["keyword_detection"] = False
        
        # If speech rate analysis is enabled, calibrate target WPM
        if self.enabled_analyses["speech_rate"]:
            print("\n--- 발화 속도 분석 설정 ---")
            print("원하는 발화 속도를 파악하기 위해 다음 문장을 읽어주세요:")
            calibration_text = "죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다. 오늘 밤에도 별이 바람에 스치운다."
            print(f"\n\"{calibration_text}\"\n")
            input("준비가 되셨으면 Enter 키를 누르고 위 문장을 읽기 시작하세요...")
            
            try:
                # Record calibration audio using audio engine
                calibration_duration = 15.0
                calibration_path = self.audio_engine.record(calibration_duration, "calibration_temp.wav")
                
                # Use Google Cloud STT for calibration
                from google.cloud import speech
                client = speech.SpeechClient()
                
                print("발화 속도를 분석 중...")
                print("Sending request to Google Cloud STT...")
                
                with open(calibration_path, "rb") as audio_file:
                    content = audio_file.read()
                
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="ko-KR",
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=True,  # Get word-level timing info
                )
                
                print("Waiting for Google Cloud STT to complete...")
                response = client.recognize(config=config, audio=audio)
                print("Received response from Google Cloud STT.")
                
                # Calculate actual speech duration from STT word timestamps (most accurate)
                # Sum up individual word durations to exclude silence between words
                actual_speech_duration = calibration_duration  # Default fallback
                
                if response.results:
                    # Collect all words with timestamps from STT results
                    all_words = []
                    
                    for result in response.results:
                        if result.alternatives and hasattr(result.alternatives[0], 'words'):
                            words = result.alternatives[0].words
                            for word in words:
                                if (hasattr(word.start_time, 'total_seconds') and 
                                    hasattr(word.end_time, 'total_seconds')):
                                    all_words.append({
                                        'word': word.word,
                                        'start': word.start_time.total_seconds(),
                                        'end': word.end_time.total_seconds()
                                    })
                    
                    # If we got word timestamps, sum up individual word durations
                    # This excludes silence between words!
                    if all_words:
                        # Calculate individual word durations and filter out anomalies
                        word_durations = []
                        for w in all_words:
                            duration = w['end'] - w['start']
                            # Sanity check: word duration should be between 0.01s and 5s
                            if 0.01 <= duration <= 5.0:
                                word_durations.append(duration)
                            else:
                                print(f"   [경고] 비정상 단어 duration 제외: '{w['word']}' ({duration:.3f}s)")
                        
                        if word_durations:
                            total_word_duration = sum(word_durations)
                            actual_speech_duration = total_word_duration
                            
                            print(f"실제 발화 시간: {actual_speech_duration:.2f}초 (STT word timestamps 기반)")
                            print(f"   단어 수: {len(word_durations)}개, 평균 단어 길이: {actual_speech_duration/len(word_durations):.3f}초")
                            
                            # Show timing span for reference
                            first_word_start = all_words[0]['start']
                            last_word_end = all_words[-1]['end']
                            total_span = last_word_end - first_word_start
                            silence_between_words = total_span - actual_speech_duration
                            print(f"   첫 단어 시작: {first_word_start:.2f}s, 마지막 단어 끝: {last_word_end:.2f}s")
                            print(f"   전체 구간: {total_span:.2f}s, 단어 간 침묵: {silence_between_words:.2f}s")
                        else:
                            # All word durations were invalid, fallback to VAD
                            print("   [경고] 모든 단어 duration이 비정상적임, VAD로 fallback")
                            from .audio_utils import detect_speech_duration
                            actual_speech_duration = detect_speech_duration(
                                calibration_path, 
                                sample_rate=16000,
                                top_db=60
                            )
                            print(f"실제 발화 시간: {actual_speech_duration:.2f}초 (VAD 기반, 부정확할 수 있음)")
                    else:
                        # Fallback to VAD-based detection
                        from .audio_utils import detect_speech_duration
                        actual_speech_duration = detect_speech_duration(
                            calibration_path, 
                            sample_rate=16000,
                            top_db=60
                        )
                        print(f"실제 발화 시간: {actual_speech_duration:.2f}초 (VAD 기반, 부정확할 수 있음)")
                
                # Final sanity check: speech duration should be reasonable
                if actual_speech_duration > calibration_duration:
                    print(f"   [경고] 발화 시간({actual_speech_duration:.2f}s)이 녹음 시간({calibration_duration:.1f}s)보다 김!")
                    print(f"   녹음 시간으로 보정합니다.")
                    actual_speech_duration = calibration_duration
                elif actual_speech_duration < 1.0:
                    print(f"   [경고] 발화 시간이 너무 짧음 ({actual_speech_duration:.2f}s)")
                    print(f"   최소값 1.0초로 보정합니다.")
                    actual_speech_duration = 1.0
                
                print(f"   (전체 녹음 시간: {calibration_duration:.1f}초)")
                
                calibration_transcript = []
                if response.results:
                    for result in response.results:
                        if result.alternatives:
                            calibration_transcript.append({
                                "text": result.alternatives[0].transcript,
                                "speaker": "SPEAKER_00",
                                "start": 0,
                                "end": actual_speech_duration,  # Use actual speech duration
                            })
                
                if calibration_transcript:
                    # Analyze speech rate
                    calibration_analysis = self.speech_rate_analyzer.analyze(calibration_transcript)
                    
                    if calibration_analysis:
                        # Calculate average WPM
                        total_word_count = sum(seg.get("word_count", 0) for seg in calibration_analysis)
                        total_duration = sum(seg.get("duration", 0) for seg in calibration_analysis)
                        
                        if total_duration > 0:
                            avg_wpm = (total_word_count / total_duration) * 60
                            self.target_wpm = avg_wpm
                            self.speech_rate_analyzer.set_target_wpm(avg_wpm)
                            print(f"\n✅ 목표 발화 속도: {avg_wpm:.2f} WPM")
                            print(f"   (이 속도를 기준으로 발화 속도를 평가합니다.)")
                        else:
                            print("⚠️ 발화 속도를 측정할 수 없습니다. 기본 설정을 사용합니다.")
                    else:
                        print("⚠️ 발화 속도를 측정할 수 없습니다. 기본 설정을 사용합니다.")
                else:
                    print("⚠️ 음성이 감지되지 않았습니다. 기본 설정을 사용합니다.")
                
                # Clean up calibration file
                if os.path.exists(calibration_path):
                    os.remove(calibration_path)
                    
            except Exception as e:
                print(f"⚠️ 발화 속도 측정 중 오류 발생: {e}")
                print("기본 설정을 사용합니다.")
        
        print("\n✅ 세션 준비 완료!")
    
    def stt_callback(self, transcript: str, is_final: bool, speaker: str, timing_info: Optional[Dict] = None):
        """
        Callback function for streaming STT results.
        
        Args:
            transcript: The transcribed text
            is_final: Whether this is a final result
            speaker: Speaker label
            timing_info: Optional dict with 'start_time', 'end_time', and 'word_timestamps'
        """
        # Only process if we're actively recording
        if not self.is_recording:
            return
            
        if not transcript:
            return
        
        # Process both interim and final results for instant detection
        # Interim results are analyzed for fast feedback (keywords, profanity, speech rate)
        # Final results are stored in transcript buffer for full analysis
        
        # Prepare timing info
        if timing_info and timing_info.get("start_time") is not None:
            audio_start_time = timing_info["start_time"]
            audio_end_time = timing_info.get("end_time", audio_start_time)
            
            if self.session_start_time is None:
                self.session_start_time = datetime.fromtimestamp(time.time() - audio_start_time)
            
            timestamp = self.session_start_time.timestamp() + audio_start_time
        else:
            timestamp = time.time()
            audio_start_time = None
            audio_end_time = None
        
        if not is_final:
            # INTERIM RESULT: Word Count Tracking approach
            # Key insight: Count how many times each keyword/profanity appears in the interim
            # Only provide feedback for NEW occurrences (count increased)
            # This is robust to STT text variations and provides instant feedback
            
            # Skip if text hasn't changed at all (exact duplicate)
            if transcript == self.last_interim_text:
                return
            
            self.last_interim_text = transcript
            
            # Analyze the current interim text to count keywords/profanity
            transcript_item = {
                "text": transcript,
                "timestamp": timestamp,
                "speaker": speaker,
                "is_final": False,
                "audio_start_time": audio_start_time,
                "audio_end_time": audio_end_time,
                "word_timestamps": timing_info.get("word_timestamps", []) if timing_info else []
            }
            
            # Store as last interim (in case session ends before Final)
            self.last_interim_item = transcript_item.copy()
            
            # Run fast analysis in separate thread with count-based feedback
            analysis_thread = threading.Thread(
                target=self._run_fast_analysis_sync,
                args=(transcript_item,),
                daemon=True
            )
            analysis_thread.start()
        
        else:
            # FINAL RESULT: Reset interim tracking and run full analysis
            # Reset word counts for next sentence
            self.last_interim_text = ""
            self.last_interim_item = None  # Clear interim since we got Final
            self.interim_word_counts.clear()  # Clear counts for next sentence
            
            # Create unique ID for this transcript to prevent duplicates
            transcript_id = f"{timestamp}_{transcript[:50]}"
            
            # Check if already processed
            if transcript_id in self.processed_transcript_ids:
                return  # Skip duplicate
            
            # Mark as processed
            self.processed_transcript_ids.add(transcript_id)
            
            # Store final transcript with timing information
            transcript_item = {
                "text": transcript,
                "timestamp": timestamp,
                "speaker": speaker,
                "is_final": True,
                "id": transcript_id,
                "audio_start_time": audio_start_time,
                "audio_end_time": audio_end_time,
                "word_timestamps": timing_info.get("word_timestamps", []) if timing_info else []
            }
            self.transcript_buffer.append(transcript_item)
            
            # Print to console immediately
            print(f"\n✅ 인식: {transcript}")
            
            # Trigger full analysis immediately
            # Run in a new thread to avoid blocking STT callback
            analysis_thread = threading.Thread(
                target=self._run_analysis_sync,
                args=(transcript_item,),
                daemon=True
            )
            analysis_thread.start()
    
    def _run_fast_analysis_sync(self, transcript_item: Dict):
        """Run fast analysis only (keywords, profanity, speech rate) for interim results."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._analyze_fast_only(transcript_item))
        finally:
            loop.close()
    
    def _run_analysis_sync(self, transcript_item: Dict):
        """Run full analysis synchronously in a separate thread."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._analyze_single_transcript(transcript_item))
        finally:
            loop.close()
    
    def _prepare_segment(self, transcript_item: Dict):
        """Prepare segment data structure from transcript item."""
        transcript_timestamp = transcript_item["timestamp"]
        text = transcript_item["text"]
        word_count = len(text.split())
        
        audio_start = transcript_item.get("audio_start_time")
        audio_end = transcript_item.get("audio_end_time")
        
        if audio_start is not None and audio_end is not None and audio_end > audio_start:
            actual_duration = audio_end - audio_start
            segment_start = transcript_timestamp
            segment_end = transcript_timestamp + actual_duration
        else:
            estimated_duration = max(0.5, (word_count / 150.0) * 60.0)
            segment_start = transcript_timestamp
            segment_end = transcript_timestamp + estimated_duration
        
        return {
            "text": text,
            "speaker": transcript_item["speaker"],
            "start": segment_start,
            "end": segment_end,
            "words": transcript_item.get("word_timestamps", [])
        }
    
    async def _analyze_fast_only(self, transcript_item: Dict):
        """Fast analysis for interim results: keywords and profanity only."""
        segment = self._prepare_segment(transcript_item)
        text = transcript_item["text"]
        
        loop = asyncio.get_running_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = []
            
            # Keyword detection (fast)
            if self.enabled_analyses["keyword_detection"] and self.custom_keywords:
                keyword_task = loop.run_in_executor(
                    pool, self.word_analyzer.analyze, [segment], self.custom_keywords
                )
                tasks.append(("keywords", keyword_task))
            
            # Profanity detection (fast)
            if self.enabled_analyses["profanity_detection"]:
                profanity_task = loop.run_in_executor(
                    pool, self.word_analyzer.analyze, [segment], self.profanity_list
                )
                tasks.append(("profanity", profanity_task))
            
            # Wait for fast analyses
            for task_name, task in tasks:
                try:
                    result = await task
                    
                    if task_name == "keywords" and result:
                        current_counts = {}
                        keyword_items = {}
                        for item in result:
                            keyword_lower = item['keyword'].lower()
                            current_counts[keyword_lower] = current_counts.get(keyword_lower, 0) + 1
                            if keyword_lower not in keyword_items:
                                keyword_items[keyword_lower] = item
                        
                        for keyword_lower, current_count in current_counts.items():
                            count_key = ("keyword", keyword_lower)
                            prev_count = self.interim_word_counts.get(count_key, 0)
                            new_occurrences = current_count - prev_count
                            
                            if new_occurrences > 0:
                                item = keyword_items[keyword_lower]
                                keyword = item['keyword']
                                
                                if "timestamp" not in item or item["timestamp"] is None:
                                    item["timestamp"] = transcript_item["timestamp"]
                                
                                # Real-time feedback
                                for i in range(new_occurrences):
                                    msg = f"키워드 검출: '{keyword}'"
                                    self.feedback_queue.put(msg)
                                    self.ui_feedback_queue.put({"message": msg, "type": "keyword"})
                                
                                # Store for report
                                report_key = ("keyword", keyword_lower, round(item["timestamp"], 1))
                                if report_key not in self.detected_items_for_report:
                                    self.detected_items_for_report.add(report_key)
                                    self.all_keyword_detections.append(item)
                                
                                self.interim_word_counts[count_key] = current_count
                    
                    elif task_name == "profanity" and result:
                        current_counts = {}
                        profanity_items = {}
                        for item in result:
                            profanity_lower = item['keyword'].lower()
                            current_counts[profanity_lower] = current_counts.get(profanity_lower, 0) + 1
                            if profanity_lower not in profanity_items:
                                profanity_items[profanity_lower] = item
                        
                        for profanity_lower, current_count in current_counts.items():
                            count_key = ("profanity", profanity_lower)
                            prev_count = self.interim_word_counts.get(count_key, 0)
                            new_occurrences = current_count - prev_count
                            
                            if new_occurrences > 0:
                                item = profanity_items[profanity_lower]
                                profanity = item['keyword']
                                
                                if "timestamp" not in item or item["timestamp"] is None:
                                    item["timestamp"] = transcript_item["timestamp"]
                                
                                # Real-time feedback
                                for i in range(new_occurrences):
                                    msg = f"비속어 검출: '{profanity}'"
                                    self.feedback_queue.put(msg)
                                    self.ui_feedback_queue.put({"message": msg, "type": "profanity"})
                                
                                # Store for report
                                report_key = ("profanity", profanity_lower, round(item["timestamp"], 1))
                                if report_key not in self.detected_items_for_report:
                                    self.detected_items_for_report.add(report_key)
                                    self.all_profanity_detections.append(item)
                                
                                self.interim_word_counts[count_key] = current_count
                
                except Exception as e:
                    print(f"⚠️ Error in {task_name} analysis: {e}")
    
    async def _analyze_single_transcript(self, transcript_item: Dict):
        """Analyze a single transcript immediately (full analysis for final results)."""
        segment = self._prepare_segment(transcript_item)
        text = transcript_item["text"]
        word_count = len(segment["text"].split())
        
        # Analyze speech rate for this Final result (per-sentence analysis)
        if self.enabled_analyses["speech_rate"]:
            self._analyze_speech_rate_for_sentence(transcript_item)
        
        # Add FINAL text to dialect buffer (for 10-second window analysis)
        if self.enabled_analyses["dialect"]:
            self.dialect_text_buffer.append(text)
        
        loop = asyncio.get_running_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = []
            
            # Keyword detection
            if self.enabled_analyses["keyword_detection"] and self.custom_keywords:
                keyword_task = loop.run_in_executor(
                    pool, self.word_analyzer.analyze, [segment], self.custom_keywords
                )
                tasks.append(("keywords", keyword_task))
            
            # Profanity detection
            if self.enabled_analyses["profanity_detection"]:
                profanity_task = loop.run_in_executor(
                    pool, self.word_analyzer.analyze, [segment], self.profanity_list
                )
                tasks.append(("profanity", profanity_task))
            
            # Speech rate (fast)
            if self.enabled_analyses["speech_rate"]:
                speech_rate_task = loop.run_in_executor(
                    pool, self.speech_rate_analyzer.analyze, [segment]
                )
                tasks.append(("speech_rate", speech_rate_task))
            
            # Wait for fast analyses
            for task_name, task in tasks:
                try:
                    result = await task
                    
                    if task_name == "keywords" and result:
                        for item in result:
                            if "timestamp" not in item or item["timestamp"] is None:
                                item["timestamp"] = transcript_item["timestamp"]
                            
                            # Store for report (with deduplication)
                            keyword_lower = item['keyword'].lower()
                            report_key = ("keyword", keyword_lower, round(item["timestamp"], 1))
                            if report_key not in self.detected_items_for_report:
                                self.detected_items_for_report.add(report_key)
                                self.all_keyword_detections.append(item)
                    
                    elif task_name == "profanity" and result:
                        for item in result:
                            if "timestamp" not in item or item["timestamp"] is None:
                                item["timestamp"] = transcript_item["timestamp"]
                            
                            # Store for report (with deduplication)
                            profanity_lower = item['keyword'].lower()
                            report_key = ("profanity", profanity_lower, round(item["timestamp"], 1))
                            if report_key not in self.detected_items_for_report:
                                self.detected_items_for_report.add(report_key)
                                self.all_profanity_detections.append(item)
                    
                    elif task_name == "speech_rate" and result:
                        # Ensure each result has proper timestamp
                        for seg in result:
                            # Use actual timestamps from segment (already set correctly above)
                            # Recalculate WPM if duration is valid
                            duration = seg.get("duration", 0)
                            if duration > 0:
                                word_count_seg = seg.get("word_count", word_count)
                                seg["wpm"] = (word_count_seg / duration) * 60
                                seg["wps"] = word_count_seg / duration
                            else:
                                # If duration is 0 or missing, recalculate from start/end
                                start = seg.get("start", segment["start"])
                                end = seg.get("end", segment["end"])
                                duration = end - start
                                if duration > 0:
                                    seg["duration"] = duration
                                    word_count_seg = seg.get("word_count", word_count)
                                    seg["wpm"] = (word_count_seg / duration) * 60
                                    seg["wps"] = word_count_seg / duration
                            
                        # Store for report (NO feedback - already given in interim)
                        self.all_speech_rate_results.extend(result)
                
                except Exception as e:
                    print(f"⚠️ Error in {task_name} analysis: {e}")
        
        # Dialect vocabulary detection
        if self.enabled_analyses["dialect"] and self.dialect_analyzer:
            detected_words = self.dialect_analyzer.detect_dialect_vocabulary(
                transcript_item["text"], 
                transcript_item.get("timestamp"), 
                is_final=True
            )
            if detected_words:
                for word_info in detected_words:
                    msg = f"방언 용어 감지: '{word_info['word']}' ({word_info['region']}도)"
                    self.feedback_queue.put(msg)
                    self.ui_feedback_queue.put({"message": msg, "type": "dialect_vocabulary"})
        
        # Slow analyses (grammar, context) - run periodically
        current_time = time.time()
        if current_time - self.last_analysis_time > 5.0:
            try:
                await self._run_slow_analysis()
            except Exception as e:
                print(f"⚠️ Slow analysis error: {e}")
            self.last_analysis_time = current_time
    
    async def _run_slow_analysis(self):
        """Run slow analyses (grammar, context)."""
        if not self.transcript_buffer:
            return
        
        # Filter out already analyzed transcripts
        new_transcripts = [
            item for item in self.transcript_buffer 
            if item["id"] not in self.llm_analyzed_transcript_ids
        ]
        
        # If no new transcripts, skip analysis
        if not new_transcripts:
            return
        
        # Mark these transcripts as analyzed
        for item in new_transcripts:
            self.llm_analyzed_transcript_ids.add(item["id"])
        
        # Convert buffer to segment format with estimated durations
        segments = []
        for item in new_transcripts:
            text = item["text"]
            word_count = len(text.split())
            # Estimate duration based on average speaking rate (~150 WPM)
            estimated_duration = max(0.5, (word_count / 150.0) * 60.0)
            
            segments.append({
                "text": text,
                "speaker": item["speaker"],
                "start": item["timestamp"],
                "end": item["timestamp"] + estimated_duration,
            })
        
        # Run LLM analysis
        if self.enabled_analyses["grammar"] or self.enabled_analyses["context"]:
            try:
                llm_results = await self.text_analyzer.analyze(segments)
                
                if self.enabled_analyses["grammar"]:
                    grammar_errors = llm_results.get("grammar_errors", [])
                    
                    # Ensure each error has proper timestamp from its segment
                    for error in grammar_errors:
                        if "timestamp" not in error or error["timestamp"] is None:
                            segment_idx = error.get("segment_index")
                            if segment_idx is not None and 0 <= segment_idx < len(segments):
                                error["timestamp"] = segments[segment_idx].get("start", time.time())
                            else:
                                error["timestamp"] = time.time()
                    
                    self.all_grammar_errors.extend(grammar_errors)
                
                if self.enabled_analyses["context"]:
                    context_errors = llm_results.get("context_errors", [])
                    
                    # Ensure each error has proper timestamp from its segment
                    for error in context_errors:
                        if "timestamp" not in error or error["timestamp"] is None:
                            segment_idx = error.get("segment_index")
                            if segment_idx is not None and 0 <= segment_idx < len(segments):
                                error["timestamp"] = segments[segment_idx].get("start", time.time())
                            else:
                                error["timestamp"] = time.time()
                    
                    self.all_context_errors.extend(context_errors)
            
            except Exception as e:
                print(f"⚠️ Error in LLM analysis: {e}")
    
    def audio_callback(self, audio_chunk: bytes):
        """
        Callback to receive audio chunks from the streaming STT.
        
        Args:
            audio_chunk: Raw audio bytes
        """
        try:
            # Always send to UI queue for visualization (even before recording starts)
            self.audio_queue.put(audio_chunk)
            
            # Only process for analysis if we're actively recording
            if not self.is_recording:
                return
            
            # Add to buffer for stutter analysis
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_buffer.extend(audio_array)
            
            # Add to speech rate buffer (for 10-second window analysis)
            if self.enabled_analyses["speech_rate"]:
                self.speech_rate_audio_buffer.extend(audio_array)
                
                # Speech rate is now analyzed per-sentence in _analyze_speech_rate_for_sentence
                # No need for periodic checks anymore
            
            # Add to dialect buffer (for 10-second window analysis)
            if self.enabled_analyses["dialect"]:
                self.dialect_audio_buffer.extend(audio_array)
                
                # Check if 10 seconds have passed since last dialect check
                current_time = time.time()
                if current_time - self.last_dialect_check >= 10.0:
                    self._check_dialect_10s_window()
                    self.last_dialect_check = current_time
            
            # Real-time stutter detection (if enabled)
            if self.enabled_analyses["stutter"] and self.stutter_detector:
                self.stutter_detector.add_audio_chunk(audio_chunk)
                
                # Check for new detections and send to UI/console
                recent_events = self.stutter_detector.get_recent_events(time_window=2.0)
                for event in recent_events:
                    # Only notify once per event (check if we've already seen it)
                    event_id = f"{event['type']}_{event['timestamp']:.1f}"
                    if not hasattr(self, '_notified_stutter_events'):
                        self._notified_stutter_events = set()
                    
                    if event_id not in self._notified_stutter_events:
                        self._notified_stutter_events.add(event_id)
                        
                        # Send feedback
                        event_type_names = {
                            'repetition': '반복',
                            'prolongation': '연장',
                            'block': '막힘'
                        }
                        event_name = event_type_names.get(event['type'], event['type'])
                        
                        feedback_msg = f"말더듬 검출 ({event_name})"
                        self.feedback_queue.put(feedback_msg)
                        self.ui_feedback_queue.put({"type": "stutter", "message": feedback_msg})
        except Exception as e:
            pass  # Silently ignore errors in callback
    
    def _analyze_speech_rate_for_sentence(self, transcript_item: Dict):
        """
        Analyze speech rate for a single sentence (Final result).
        This provides per-sentence feedback which is more natural and accurate.
        
        Args:
            transcript_item: Dict with 'text', 'timestamp', 'word_timestamps'
        """
        try:
            word_timestamps = transcript_item.get("word_timestamps", [])
            
            if not word_timestamps:
                return
            
            # Calculate actual speech duration by summing individual word durations
            word_durations = []
            for word_info in word_timestamps:
                if 'start' in word_info and 'end' in word_info:
                    duration = word_info['end'] - word_info['start']
                    # Sanity check: word duration should be between 0.01s and 5s
                    if 0.01 <= duration <= 5.0:
                        word_durations.append(duration)
            
            # Need at least 5 words for meaningful sentence-level analysis
            if len(word_durations) < 5:
                return
            
            # Total speech duration = sum of all word durations (excluding pauses)
            speech_duration = sum(word_durations)
            word_count = len(word_durations)
            
            # Calculate WPM
            if speech_duration >= 0.5 and word_count > 0:
                wpm = (word_count / speech_duration) * 60.0
                
                # Sanity check: WPM should be between 30 and 300
                if 30 <= wpm <= 300 and self.target_wpm is not None:
                    tolerance = self.target_wpm * 0.2
                    
                    if wpm > self.target_wpm + tolerance:
                        msg = f"발화 속도가 빠릅니다: {wpm:.0f} WPM (목표: {self.target_wpm:.0f})"
                        print(f"🔔 {msg}")
                        self.feedback_queue.put(msg)
                        self.ui_feedback_queue.put({"message": msg, "type": "speech_rate"})
                    elif wpm < self.target_wpm - tolerance:
                        msg = f"발화 속도가 느립니다: {wpm:.0f} WPM (목표: {self.target_wpm:.0f})"
                        print(f"🔔 {msg}")
                        self.feedback_queue.put(msg)
                        self.ui_feedback_queue.put({"message": msg, "type": "speech_rate"})
            
        except Exception as e:
            print(f"⚠️ Error in speech rate analysis: {e}")
    
    def _check_dialect_10s_window(self):
        """
        Check dialect for the past 10-second window.
        Combines acoustic analysis (Wav2Vec2) and vocabulary analysis.
        Provides real-time feedback if non-standard dialect is detected.
        """
        try:
            # Check if we have enough audio data
            if len(self.dialect_audio_buffer) < 16000 * 3:  # At least 3 seconds
                return
            
            # Get text from buffer
            combined_text = " ".join(self.dialect_text_buffer) if self.dialect_text_buffer else None
            
            # Need at least some text or audio for analysis
            if not combined_text and len(self.dialect_audio_buffer) < 16000 * 5:
                self.dialect_text_buffer.clear()
                return
            
            # Convert audio buffer to numpy array
            audio_data = np.array(self.dialect_audio_buffer, dtype=np.float32) / 32768.0
            
            # Analyze segment (acoustic + vocabulary)
            if self.dialect_analyzer and self.dialect_analyzer.is_available():
                result = self.dialect_analyzer.analyze_segment_realtime(
                    audio_array=audio_data,
                    sample_rate=16000,
                    timestamp=time.time(),
                    text=combined_text
                )
                
                # Provide real-time feedback if trigger is set
                if result.get('feedback_trigger', False):
                    verdict = result.get('combined_verdict')
                    confidence = result.get('confidence', 0.0)
                    
                    # Acoustic-based feedback
                    if result.get('acoustic_analysis') and result['acoustic_analysis']['verdict'] == 'non_standard':
                        msg = f"방언 억양 감지됨 (신뢰도: {confidence*100:.0f}%)"
                        print(f"🔔 {msg}")
                        self.feedback_queue.put(msg)
                        self.ui_feedback_queue.put({"message": msg, "type": "dialect_acoustic"})
                    
                    # Vocabulary feedback is already handled in _analyze_fast_only
            
            # Clear text buffer for next window
            # (audio buffer is a deque with maxlen, so it auto-manages)
            self.dialect_text_buffer.clear()
            
        except Exception as e:
            print(f"⚠️ Error in dialect window check: {e}")
    
    def streaming_producer(self):
        """
        Streaming producer: captures audio and sends to GCP STT.
        Starts in paused mode for instant detection when triggered.
        """
        print("🎤 Streaming producer started")
        
        try:
            # Initialize streaming STT with callbacks
            self.streaming_stt = GoogleSTTStreaming(
                callback=self.stt_callback,
                audio_callback=self.audio_callback
            )
            
            # Pre-initialize audio stream for instant detection
            self.streaming_stt.prepare()
            
            # Start streaming (blocking, but paused until resume() is called)
            # This keeps connection warm for instant detection
            self.streaming_stt.start_streaming()
            
        except Exception as e:
            print(f"❌ Error in streaming producer: {e}")
        finally:
            print("🎤 Streaming producer stopped")
    
    def console_feedback_loop(self):
        """Console feedback loop in a separate thread."""
        print("\n" + "="*60)
        print("🎙️ 세션 시작")
        print("="*60)
        print("실시간 피드백이 아래에 표시됩니다.")
        print("세션을 종료하려면 UI 창을 닫거나 Ctrl+C를 누르세요.")
        print("="*60 + "\n")
        
        while not self.stop_event.is_set():
            try:
                # Get feedback from queue with timeout
                feedback = self.feedback_queue.get(timeout=0.1)
                print(f"🔔 {feedback}")
            except Empty:
                continue
    
    def run_ui_main_thread(self):
        """Run the pygame UI visualizer in the MAIN thread (required for macOS)."""
        try:
            from .ui_visualizer import VoiceVisualizer
            
            print("🎨 Starting UI visualizer...")
            visualizer = VoiceVisualizer(self.audio_queue, self.ui_feedback_queue)
            visualizer.run()
            
            # When UI closes, stop the session
            if not self.stop_event.is_set():
                print("\n⏸️ UI closed. Stopping session...")
                self.stop_event.set()
                if self.streaming_stt:
                    self.streaming_stt.stop_streaming()
        
        except Exception as e:
            print(f"❌ Error in UI: {e}")
            self.stop_event.set()
    
    def generate_summary_report(self):
        """Generate and print a comprehensive summary report after the session ends."""
        import os
        import tempfile
        import soundfile as sf
        
        print("\n\n" + "="*60)
        print("📋 세션 요약 리포트")
        print("="*60)
        
        if not self.transcript_buffer:
            print("\n분석할 데이터가 없습니다.")
            return
        
        # Show all transcripts
        print("\n--- ✅ 전체 대화 내용 ---")
        for item in self.transcript_buffer:
            speaker = item.get("speaker", "UNKNOWN")
            text = item.get("text", "")
            timestamp = item.get("timestamp", 0)
            print(f"[{timestamp:.2f}s] {speaker}: {text}")
        
        # Keyword detection summary
        if self.enabled_analyses["keyword_detection"]:
            print("\n--- 🔍 키워드 검출 요약 ---")
            if self.all_keyword_detections:
                keyword_counts = defaultdict(int)
                for item in self.all_keyword_detections:
                    keyword_counts[item["keyword"].lower()] += 1
                
                print(f"총 {len(self.all_keyword_detections)}회 검출:")
                for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - '{keyword}': {count}회")
            else:
                print("검출된 키워드가 없습니다.")
        
        # Profanity detection summary
        if self.enabled_analyses["profanity_detection"]:
            print("\n--- ⚠️ 비속어 검출 요약 ---")
            if self.all_profanity_detections:
                profanity_counts = defaultdict(int)
                for item in self.all_profanity_detections:
                    profanity_counts[item["keyword"].lower()] += 1
                
                print(f"총 {len(self.all_profanity_detections)}회 검출:")
                for profanity, count in sorted(profanity_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - '{profanity}': {count}회")
            else:
                print("검출된 비속어가 없습니다.")
        
        # Speech rate summary
        if self.enabled_analyses["speech_rate"]:
            print("\n--- 🏃 발화 속도 분석 요약 ---")
            if self.all_speech_rate_results:
                total_word_count = sum(seg.get("word_count", 0) for seg in self.all_speech_rate_results)
                total_duration = sum(seg.get("duration", 0) for seg in self.all_speech_rate_results)
                
                if total_duration > 0:
                    overall_wpm = (total_word_count / total_duration) * 60
                    print(f"전체 평균 발화 속도: {overall_wpm:.2f} WPM")
                    
                    if self.target_wpm:
                        print(f"목표 발화 속도: {self.target_wpm:.2f} WPM")
                        
                        too_fast = sum(1 for seg in self.all_speech_rate_results if seg.get("comparison") == "too_fast")
                        too_slow = sum(1 for seg in self.all_speech_rate_results if seg.get("comparison") == "too_slow")
                        good = sum(1 for seg in self.all_speech_rate_results if seg.get("comparison") == "good")
                        
                        print(f"\n발화 속도 분포:")
                        print(f"  - 적절: {good}회")
                        print(f"  - 너무 빠름: {too_fast}회")
                        print(f"  - 너무 느림: {too_slow}회")
            else:
                print("발화 속도 분석 결과가 없습니다.")
        
        # Grammar analysis summary
        if self.enabled_analyses["grammar"]:
            print("\n--- 🧐 문법 분석 요약 ---")
            if self.all_grammar_errors:
                print(f"총 {len(self.all_grammar_errors)}개의 문법 오류 발견:")
                for i, error in enumerate(self.all_grammar_errors[:10], 1):  # Show first 10
                    details = error.get("error_details", {})
                    print(f"\n  {i}. [{error.get('speaker')}] '{details.get('original')}' → '{details.get('corrected')}'")
                    print(f"     설명: {details.get('explanation')}")
                
                if len(self.all_grammar_errors) > 10:
                    print(f"\n  ... 그 외 {len(self.all_grammar_errors) - 10}개 더")
            else:
                print("문법 오류가 발견되지 않았습니다.")
        
        # Context analysis summary
        if self.enabled_analyses["context"]:
            print("\n--- 🧠 맥락 분석 요약 ---")
            if self.all_context_errors:
                print(f"총 {len(self.all_context_errors)}개의 맥락 오류 발견:")
                for i, error in enumerate(self.all_context_errors[:5], 1):  # Show first 5
                    utterance = error.get('utterance', '')
                    # Truncate long utterances for readability
                    if len(utterance) > 150:
                        utterance = utterance[:150] + "..."
                    print(f"\n  {i}. [{error.get('speaker')}] \"{utterance}\"")
                    print(f"     분석: {error.get('reasoning')}")
                
                if len(self.all_context_errors) > 5:
                    print(f"\n  ... 그 외 {len(self.all_context_errors) - 5}개 더")
            else:
                print("맥락 오류가 발견되지 않았습니다.")
        
        # Stutter analysis summary
        if self.enabled_analyses["stutter"]:
            print("\n--- 🗣️ 말더듬 분석 요약 ---")
            
            # Get real-time detection results first
            realtime_events = []
            if self.stutter_detector:
                realtime_events = self.stutter_detector.get_detected_events()
                realtime_stats = self.stutter_detector.get_statistics()
                
                if realtime_events:
                    print(f"\n✨ 실시간 오디오 분석 결과 (STT 변환 전 원본 오디오 기반):")
                    print(f"총 {realtime_stats['total_events']}개의 말더듬 이벤트 실시간 검출")
                    print(f"  • 반복: {realtime_stats['repetitions']}회")
                    print(f"  • 연장: {realtime_stats['prolongations']}회")
                    print(f"  • 막힘: {realtime_stats['blocks']}회")
                    
                    # Show some examples
                    print("\n  최근 검출 예시:")
                    for event in realtime_events[-5:]:  # Last 5 events
                        event_type_names = {
                            'repetition': '반복',
                            'prolongation': '연장',
                            'block': '막힘'
                        }
                        event_name = event_type_names.get(event['type'], event['type'])
                        duration_info = f" ({event['duration']}초)" if 'duration' in event else ""
                        print(f"  - {event_name}{duration_info} (신뢰도: {event.get('confidence', 'N/A')})")
            
            # Run stutter analysis if enabled and we have audio buffer
            if len(self.audio_buffer) > 0:
                try:
                    # Save audio buffer to temporary file
                    import tempfile
                    import soundfile as sf
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        # Convert deque to numpy array
                        audio_array = np.array(list(self.audio_buffer), dtype=np.int16)
                        # Save as WAV file
                        sf.write(temp_audio_path, audio_array, 16000)
                        
                        print("\n📊 텍스트 기반 분석 (STT 변환 후):")
                        
                        # Convert transcript buffer to segment format
                        segments = []
                        for item in self.transcript_buffer:
                            segments.append({
                                "text": item["text"],
                                "speaker": item["speaker"],
                                "start": item["timestamp"],
                                "end": item["timestamp"] + 2.0,  # Estimate
                            })
                        
                        # Run stutter analysis
                        self.stutter_results = self.stutter_analyzer.analyze(temp_audio_path, segments)
                        
                        # Display formatted summary
                        if self.stutter_results:
                            stats = self.stutter_results.get("statistics", {})
                            repetitions = self.stutter_results.get("repetitions", [])
                            prolongations = self.stutter_results.get("prolongations", [])
                            blocks = self.stutter_results.get("blocks", [])
                            
                            fluency = stats.get("fluency_percentage", 0)
                            total_events = stats.get("total_events", 0)
                            
                            print(f"\n유창성 점수: {fluency:.1f}%")
                            print(f"총 {total_events}개의 말더듬 이벤트 검출")
                            
                            if repetitions:
                                print(f"\n🔁 반복 (Repetitions): {len(repetitions)}회")
                                
                                # Count by type
                                type_counts = {}
                                for rep in repetitions:
                                    rep_type = rep.get('type', 'repetition')
                                    type_counts[rep_type] = type_counts.get(rep_type, 0) + 1
                                
                                # Show breakdown
                                type_names = {
                                    'repetition': '단어 반복',
                                    'partial_repetition': '부분 반복',
                                    'sound_repetition': '음소 반복',
                                    'multiple_repetition': '다중 반복',
                                    'word_repetition': '연속 단어 반복'
                                }
                                
                                for rep_type, count in type_counts.items():
                                    type_name = type_names.get(rep_type, rep_type)
                                    print(f"  • {type_name}: {count}회")
                                
                                print("\n  예시:")
                                for rep in repetitions[:5]:
                                    print(f"  - [{rep.get('timestamp', 0):.1f}s] '{rep.get('full_match')}' (타입: {rep.get('type', 'N/A')})")
                                if len(repetitions) > 5:
                                    print(f"  ... 그 외 {len(repetitions) - 5}회 더")
                            
                            if prolongations:
                                print(f"\n⏱️ 연장 (Prolongations): {len(prolongations)}회")
                                for prol in prolongations[:3]:
                                    print(f"  - [{prol.get('timestamp', 0):.1f}s] '{prol.get('word')}' ({prol.get('duration')}초)")
                                if len(prolongations) > 3:
                                    print(f"  ... 그 외 {len(prolongations) - 3}회 더")
                            
                            if blocks:
                                print(f"\n🚫 막힘 (Blocks): {len(blocks)}회")
                                for block in blocks[:3]:
                                    print(f"  - [{block.get('timestamp', 0):.1f}s] {block.get('duration')}초 침묵")
                                if len(blocks) > 3:
                                    print(f"  ... 그 외 {len(blocks) - 3}회 더")
                            
                            if total_events == 0:
                                print("\n✅ 말더듬 이벤트가 감지되지 않았습니다. 유창한 발화입니다!")
                        
                        # Clean up temp file
                        import os
                        os.remove(temp_audio_path)
                        
                except Exception as e:
                    print(f"❌ 말더듬 분석 중 오류 발생: {e}")
            else:
                print("분석할 오디오 데이터가 없습니다.")
        
        # Dialect analysis summary (Binary classification)
        if self.enabled_analyses["dialect"]:
            print("\n--- 🗣️ 방언 분석 요약 (표준어 vs 비표준어) ---")
            
            if self.dialect_analyzer and self.dialect_analyzer.is_available():
                # 1. Real-time analysis summary
                print("\n📊 실시간 분석 요약 (10초 단위 세그먼트 분석):")
                realtime_summary = self.dialect_analyzer.get_realtime_summary()
                
                if realtime_summary['total_segments'] > 0:
                    total_segments = realtime_summary['total_segments']
                    non_standard_count = realtime_summary['non_standard_count']
                    non_standard_ratio = realtime_summary['non_standard_ratio']
                    vocab_count = realtime_summary['vocabulary_detections']
                    
                    print(f"  • 분석된 세그먼트: {total_segments}개")
                    print(f"  • 비표준어 검출: {non_standard_count}개 ({non_standard_ratio*100:.1f}%)")
                    print(f"  • 방언 어휘 검출: {vocab_count}개")
                    
                    # Show detected vocabulary
                    if vocab_count > 0:
                        detected_words = realtime_summary['detected_words']
                        word_counts = defaultdict(int)
                        region_counts = defaultdict(int)
                        
                        for word_info in detected_words:
                            word_counts[word_info['word']] += 1
                            region_counts[word_info['region']] += 1
                        
                        print(f"\n  🔍 검출된 방언 용어:")
                        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                            region = self.dialect_analyzer.dialect_vocab[word][0]
                            meaning = self.dialect_analyzer.dialect_vocab[word][1]
                            print(f"    • '{word}' ({region}도, 뜻: {meaning}): {count}회")
                        
                        if len(word_counts) > 10:
                            print(f"    ... 그 외 {len(word_counts) - 10}개 더")
                        
                        print(f"\n  📍 지역별 방언 분포:")
                        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                            print(f"    • {region}도: {count}회")
                    
                    # Real-time verdict
                    print(f"\n  ✨ 실시간 분석 판정:")
                    if non_standard_ratio >= 0.3:  # 30% 이상이 방언
                        print(f"    ⚠️ 비표준어 (방언 비율: {non_standard_ratio*100:.1f}%)")
                        print(f"    → 세션 중 방언 특성이 자주 감지되었습니다.")
                    else:
                        print(f"    ✅ 표준어 (방언 비율: {non_standard_ratio*100:.1f}%)")
                        print(f"    → 대체로 표준어를 사용하셨습니다.")
                else:
                    print("  실시간 분석 데이터가 없습니다.")
                
                # 2. Full session analysis (re-analysis for accuracy)
                if len(self.audio_buffer) > 0:
                    try:
                        import tempfile
                        import soundfile as sf
                        
                        # Convert deque to numpy array
                        audio_array = np.array(list(self.audio_buffer), dtype=np.int16)
                        total_duration = len(audio_array) / 16000
                        
                        print("\n\n📊 전체 세션 재분석 (최종 정밀 판정):")
                        print(f"  • 전체 오디오 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
                        
                        # Split long audio into segments (2 minutes each for optimal analysis)
                        segment_duration = 120  # 2 minutes in seconds
                        segment_samples = 16000 * segment_duration  # 2 minutes at 16kHz
                        
                        if len(audio_array) > segment_samples:
                            # Audio is longer than segment_duration, split it
                            num_segments = (len(audio_array) + segment_samples - 1) // segment_samples
                            print(f"  • 긴 오디오 감지: {num_segments}개 세그먼트로 분할하여 분석")
                            
                            segment_results = []
                            
                            for i in range(num_segments):
                                start_idx = i * segment_samples
                                end_idx = min((i + 1) * segment_samples, len(audio_array))
                                segment = audio_array[start_idx:end_idx]
                                
                                # Skip very short segments (< 5 seconds)
                                if len(segment) < 16000 * 5:
                                    continue
                                
                                # Save segment to temporary file
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                                    temp_audio_path = temp_audio.name
                                    sf.write(temp_audio_path, segment, 16000)
                                    
                                    print(f"    [{i+1}/{num_segments}] 세그먼트 분석 중 ({len(segment)/16000:.1f}초)...")
                                    
                                    # Analyze segment
                                    seg_classification = self.dialect_analyzer.get_classification(temp_audio_path)
                                    
                                    # Clean up temp file
                                    try:
                                        os.remove(temp_audio_path)
                                    except:
                                        pass
                                    
                                    if "error" not in seg_classification:
                                        segment_results.append(seg_classification)
                            
                            # Aggregate results from all segments
                            if segment_results:
                                # Average probabilities across segments
                                avg_standard = sum(r['probabilities']['standard'] for r in segment_results) / len(segment_results)
                                avg_non_standard = sum(r['probabilities']['non_standard'] for r in segment_results) / len(segment_results)
                                
                                is_standard = avg_standard > avg_non_standard
                                confidence = max(avg_standard, avg_non_standard)
                                
                                classification = {
                                    'is_standard': is_standard,
                                    'confidence': confidence,
                                    'probabilities': {
                                        'standard': avg_standard,
                                        'non_standard': avg_non_standard
                                    },
                                    'num_segments_analyzed': len(segment_results)
                                }
                                
                                print(f"  ✅ 분석 완료: {len(segment_results)}개 세그먼트의 평균 결과")
                            else:
                                classification = {"error": "모든 세그먼트 분석 실패"}
                        else:
                            # Audio is short enough, analyze as a single file
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                                temp_audio_path = temp_audio.name
                                sf.write(temp_audio_path, audio_array, 16000)
                                
                                print(f"  • 단일 파일로 분석 중...")
                                
                                # Get binary classification result
                                classification = self.dialect_analyzer.get_classification(temp_audio_path)
                                
                                # Clean up temp file
                                try:
                                    os.remove(temp_audio_path)
                                except Exception as cleanup_error:
                                    print(f"  ⚠️ 임시 파일 삭제 실패 (무시됨): {cleanup_error}")
                            
                        if "error" not in classification:
                            # Store for report
                            self.dialect_results = classification
                            self.dialect_results['realtime_summary'] = realtime_summary
                            
                            # Extract probabilities
                            probs = classification.get("probabilities", {})
                            standard_prob = probs.get("standard", 0.0)
                            non_standard_prob = probs.get("non_standard", 0.0)
                            is_standard = classification.get("is_standard", False)
                            confidence = classification.get("confidence", 0.0)
                            num_segments = classification.get("num_segments_analyzed")
                            
                            # Display results with bar chart
                            print("\n  확률 분포:")
                            if num_segments:
                                print(f"  ({num_segments}개 세그먼트 평균)")
                            
                            # Standard
                            bar_length_std = int(standard_prob * 50)
                            bar_std = "█" * bar_length_std + "░" * (50 - bar_length_std)
                            print(f"    표준어      [{bar_std}] {standard_prob*100:.2f}%")
                            
                            # Non-standard
                            bar_length_non = int(non_standard_prob * 50)
                            bar_non = "█" * bar_length_non + "░" * (50 - bar_length_non)
                            print(f"    비표준어    [{bar_non}] {non_standard_prob*100:.2f}%")
                            
                            # Final verdict
                            verdict = "✅ 표준어" if is_standard else "⚠️ 비표준어"
                            print(f"\n  ✨ 최종 판정: {verdict} (신뢰도: {confidence*100:.2f}%)")
                            
                            # Additional info
                            if is_standard:
                                print("    → 전체적으로 표준어 발음을 사용하고 있습니다.")
                            else:
                                print("    → 전체적으로 방언 특성이 감지되었습니다.")
                            
                            # Compare with real-time analysis
                            print(f"\n  📈 실시간 vs 전체 분석 비교:")
                            realtime_verdict = "비표준어" if realtime_summary['non_standard_ratio'] >= 0.3 else "표준어"
                            final_verdict = "비표준어" if not is_standard else "표준어"
                            
                            if realtime_verdict == final_verdict:
                                print(f"    ✓ 일치: 두 분석 모두 '{final_verdict}' 판정")
                                print(f"      신뢰도가 높은 결과입니다.")
                            else:
                                print(f"    ⚠ 불일치: 실시간({realtime_verdict}) vs 전체({final_verdict})")
                                print(f"      전체 분석 결과를 더 신뢰할 수 있습니다.")
                        else:
                            print(f"  ❌ 방언 분석 실패: {classification['error']}")
                    
                    except Exception as e:
                        print(f"❌ 방언 분석 중 오류 발생: {e}")
                else:
                    print("\n분석할 오디오 데이터가 없습니다.")
            else:
                print("방언 분석 모델이 로드되지 않았습니다.")
        
        print("\n" + "="*60)
        print("세션 종료")
        print("="*60)
        
        # Generate PDF report
        print("\n📄 PDF 리포트 생성 중...")
        try:
            session_data = {
                "session_start_time": self.session_start_time,
                "session_end_time": datetime.now(),
                "enabled_analyses": self.enabled_analyses,
                "transcripts": self.transcript_buffer,
                "keyword_detections": self.all_keyword_detections,
                "profanity_detections": self.all_profanity_detections,
                "speech_rate_results": self.all_speech_rate_results,
                "grammar_errors": self.all_grammar_errors,
                "context_errors": self.all_context_errors,
                "stutter_results": self.stutter_results,
                "stutter_detector_events": self.stutter_detector.get_detected_events() if self.stutter_detector else [],
                "stutter_detector_stats": self.stutter_detector.get_statistics() if self.stutter_detector else {},
                "dialect_results": self.dialect_results,  # Add dialect results
                "custom_keywords": self.custom_keywords,
                "target_wpm": self.target_wpm
            }
            
            pdf_path = self.report_generator.generate_report(session_data)
            print(f"✅ PDF 리포트 생성 완료: {pdf_path}")
        except Exception as e:
            print(f"❌ PDF 리포트 생성 실패: {e}")
    
    def _wait_for_start_trigger(self):
        """
        Wait for 's' key press to start recording.
        Uses a simple blocking input for cross-platform compatibility.
        """
        while True:
            try:
                key = input("\n입력: ").strip().lower()
                if key == 's':
                    break
                else:
                    print("'s' 키를 눌러주세요.", end='', flush=True)
            except KeyboardInterrupt:
                # Allow Ctrl+C to exit
                raise
    
    def run(self, enable_ui: bool = True):
        """Run the main HabitLink session."""
        # Check if initialization succeeded
        if not self.is_initialized:
            print("❌ 초기화 실패. 프로그램을 종료합니다.")
            return
        
        # Select analyses
        self.select_analyses()
        
        # Check if at least one analysis is enabled
        if not any(self.enabled_analyses.values()):
            print("\n⚠️ 활성화된 분석이 없습니다. 프로그램을 종료합니다.")
            return
        
        # Prepare session (calibration, etc.)
        self.prepare_session()
        
        # Start streaming thread (but not recording yet)
        # This pre-establishes the STT connection for instant detection
        streaming_thread = threading.Thread(target=self.streaming_producer, daemon=True)
        streaming_thread.start()
        
        # Wait for streaming to be fully ready
        print("\n⏳ 스트리밍 연결 중... (음성 인식 엔진 준비)")
        time.sleep(2.5)  # Give streaming time to establish connection and fill buffer
        
        # Show trigger prompt
        print("\n" + "="*60)
        print("🎯 모든 준비가 완료되었습니다!")
        print("="*60)
        print("\n📝 's' 키를 눌러 녹음을 시작하세요.")
        print("   (녹음을 종료하려면 UI 창을 닫거나 Ctrl+C를 누르세요)")
        print("\n대기 중...", end='', flush=True)
        
        # Wait for 's' key press to start recording
        self._wait_for_start_trigger()
        
        # Start recording - resume STT streaming
        self.is_recording = True
        if self.streaming_stt:
            self.streaming_stt.resume()  # Resume STT streaming for instant detection
        
        self.session_start_time = datetime.now()
        print(f"\n\n🔴 녹음 시작! ({self.session_start_time.strftime('%H:%M:%S')})")
        print("="*60)
        
        # Start console feedback thread
        console_thread = threading.Thread(target=self.console_feedback_loop, daemon=True)
        console_thread.start()
        
        try:
            if enable_ui:
                # Run UI in MAIN thread (required for macOS)
                self.run_ui_main_thread()
            else:
                # Console-only mode: wait for keyboard interrupt
                print("\n콘솔 모드로 실행 중...")
                print("종료하려면 Ctrl+C를 누르세요.\n")
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n⏸️ 세션을 종료하는 중...")
            self.stop_event.set()
            if self.streaming_stt:
                self.streaming_stt.stop_streaming()
        
        # Wait for threads to finish with longer timeout to ensure all STT results are processed
        print("마지막 분석을 완료하는 중...")
        time.sleep(2)  # Give STT a moment to finish processing any remaining audio
        streaming_thread.join(timeout=10)  # Increased from 5 to 10 seconds
        console_thread.join(timeout=3)  # Increased from 2 to 3 seconds
        
        # Handle last interim if session ended before Final result
        if self.last_interim_item is not None:
            # Check if this text was already saved as Final (to avoid duplicates)
            last_text = self.last_interim_item["text"]
            is_duplicate = any(
                item["text"] == last_text 
                for item in self.transcript_buffer
            )
            
            if not is_duplicate and last_text.strip():
                # Save this interim as Final
                timestamp = self.last_interim_item["timestamp"]
                transcript_id = f"{timestamp}_{last_text[:50]}"
                
                final_item = {
                    "text": last_text,
                    "timestamp": timestamp,
                    "speaker": self.last_interim_item["speaker"],
                    "is_final": True,
                    "id": transcript_id,
                    "audio_start_time": self.last_interim_item.get("audio_start_time"),
                    "audio_end_time": self.last_interim_item.get("audio_end_time"),
                    "word_timestamps": self.last_interim_item.get("word_timestamps", [])
                }
                
                self.transcript_buffer.append(final_item)
                print(f"✅ 마지막 발화 저장: {last_text[:100]}...")
        
        # Generate summary report
        self.generate_summary_report()

