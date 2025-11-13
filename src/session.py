import os
import time
import asyncio
import threading
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional
import concurrent.futures
import traceback
import numpy as np

from .audio_engine import AudioEngine
from .stt import GoogleSTTStreaming
from .word_analyzer import WordAnalyzer
from .speech_rate_analyzer import SpeechRateAnalyzer
from .text_analyzer import TextAnalyzer
from .stutter_analyzer import StutterAnalyzer
from .stutter_detector import StutterDetector
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
        self.profanity_list = []
        
        # User configuration
        self.enabled_analyses = {
            "keyword_detection": False,
            "profanity_detection": False,
            "speech_rate": False,
            "grammar": False,
            "context": False,
            "stutter": False
        }
        self.custom_keywords = []
        self.target_wpm = None
        
        # Threading components
        self.feedback_queue = Queue()
        self.ui_feedback_queue = Queue()
        self.audio_queue = Queue()
        self.stop_event = threading.Event()
        
        # Streaming buffers
        self.transcript_buffer = []  # Buffer for recent transcripts
        self.audio_buffer = deque(maxlen=16000 * 30)  # 30 seconds of audio at 16kHz
        self.last_analysis_time = time.time()
        
        # Store analysis results for summary
        self.all_keyword_detections = []
        self.all_profanity_detections = []
        self.all_speech_rate_results = []
        self.all_grammar_errors = []
        self.all_context_errors = []
        self.stutter_results = None
        
        # Track processed transcripts to avoid duplicates
        self.processed_transcript_ids = set()
        
        # Session metadata
        self.session_start_time = None
        
        # Report generator
        self.report_generator = ReportGenerator()
        
    def initialize_components(self):
        """Initialize all analysis components."""
        print("\n🚀 Initializing HabitLink components...")
        
        try:
            # Initialize audio engine (still used for calibration)
            self.audio_engine = AudioEngine(samplerate=16000, channels=1)
            print("✅ Audio engine initialized")
            
            # Initialize analyzers
            self.word_analyzer = WordAnalyzer()
            self.speech_rate_analyzer = SpeechRateAnalyzer()
            self.text_analyzer = TextAnalyzer()
            self.stutter_analyzer = StutterAnalyzer()
            self.stutter_detector = StutterDetector()
            print("✅ Analysis modules initialized")
            
            # Load profanity list
            self.profanity_list = load_profanity_list()
            print(f"✅ Profanity list loaded ({len(self.profanity_list)} words)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            traceback.print_exc()
            return False
    
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
        print("\n여러 개를 선택하려면 쉼표로 구분하세요 (예: 1,3,4)")
        
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
                )
                
                print("Waiting for Google Cloud STT to complete...")
                response = client.recognize(config=config, audio=audio)
                print("Received response from Google Cloud STT.")
                
                # Convert response to segments format
                calibration_transcript = []
                if response.results:
                    for result in response.results:
                        if result.alternatives:
                            calibration_transcript.append({
                                "text": result.alternatives[0].transcript,
                                "speaker": "SPEAKER_00",
                                "start": 0,
                                "end": 15,
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
        if not transcript:
            return
        
        if is_final:
            # Use actual audio timestamps if available, otherwise fall back to current time
            if timing_info and timing_info.get("start_time") is not None:
                # Use actual audio start time (relative to session start)
                audio_start_time = timing_info["start_time"]
                audio_end_time = timing_info.get("end_time", audio_start_time)
                
                # Store session start time if not set
                if self.session_start_time is None:
                    # Calculate session start: current time - audio_start_time
                    self.session_start_time = datetime.fromtimestamp(time.time() - audio_start_time)
                
                # Calculate absolute timestamp
                timestamp = self.session_start_time.timestamp() + audio_start_time
            else:
                # Fallback to current time
                timestamp = time.time()
                audio_start_time = None
                audio_end_time = None
            
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
                "audio_start_time": audio_start_time,  # Relative to stream start
                "audio_end_time": audio_end_time,  # Relative to stream start
                "word_timestamps": timing_info.get("word_timestamps", []) if timing_info else []
            }
            self.transcript_buffer.append(transcript_item)
            
            # Print to console immediately
            print(f"\n✅ 인식: {transcript}")
            
            # Trigger analysis immediately (don't wait for 2 seconds)
            # Run in a new thread to avoid blocking STT callback
            analysis_thread = threading.Thread(
                target=self._run_analysis_sync,
                args=(transcript_item,),
                daemon=True
            )
            analysis_thread.start()
    
    def _run_analysis_sync(self, transcript_item: Dict):
        """Run analysis synchronously in a separate thread."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._analyze_single_transcript(transcript_item))
        finally:
            loop.close()
    
    async def _analyze_single_transcript(self, transcript_item: Dict):
        """Analyze a single transcript immediately."""
        # Get actual timestamp from transcript
        transcript_timestamp = transcript_item["timestamp"]
        
        text = transcript_item["text"]
        word_count = len(text.split())
        
        # Use actual audio timestamps if available (from STT word-level timestamps)
        audio_start = transcript_item.get("audio_start_time")
        audio_end = transcript_item.get("audio_end_time")
        
        if audio_start is not None and audio_end is not None and audio_end > audio_start:
            # Use actual duration from audio timestamps
            actual_duration = audio_end - audio_start
            segment_start = transcript_timestamp
            segment_end = transcript_timestamp + actual_duration
        else:
            # Fallback: Estimate duration based on word count (assuming ~150 WPM average speaking rate)
            # This is less accurate but better than nothing
            estimated_duration = max(0.5, (word_count / 150.0) * 60.0)
            segment_start = transcript_timestamp
            segment_end = transcript_timestamp + estimated_duration
        
        # Convert to segment format with proper timestamps
        segment = {
            "text": text,
            "speaker": transcript_item["speaker"],
            "start": segment_start,
            "end": segment_end,
            "words": transcript_item.get("word_timestamps", [])  # Include word-level timestamps if available
        }
        
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
                            # Ensure timestamp is set correctly
                            if "timestamp" not in item or item["timestamp"] is None:
                                item["timestamp"] = transcript_item["timestamp"]
                            msg = f"키워드 검출: '{item['keyword']}'"
                            print(f"🔔 {msg}")
                            self.feedback_queue.put(msg)
                            self.ui_feedback_queue.put({"message": msg, "type": "keyword"})
                            # Store for summary
                            self.all_keyword_detections.append(item)
                    
                    elif task_name == "profanity" and result:
                        for item in result:
                            # Ensure timestamp is set correctly
                            if "timestamp" not in item or item["timestamp"] is None:
                                item["timestamp"] = transcript_item["timestamp"]
                            msg = f"비속어 검출: '{item['keyword']}'"
                            print(f"🔔 {msg}")
                            self.feedback_queue.put(msg)
                            self.ui_feedback_queue.put({"message": msg, "type": "profanity"})
                            # Store for summary
                            self.all_profanity_detections.append(item)
                    
                    elif task_name == "speech_rate" and result:
                        # Ensure each result has proper timestamp
                        for seg in result:
                            # Use actual timestamps from segment (already set correctly above)
                            # Recalculate WPM if duration is valid
                            duration = seg.get("duration", 0)
                            if duration > 0:
                                word_count = seg.get("word_count", word_count)
                                seg["wpm"] = (word_count / duration) * 60
                                seg["wps"] = word_count / duration
                            else:
                                # If duration is 0 or missing, recalculate from start/end
                                start = seg.get("start", segment_start)
                                end = seg.get("end", segment_end)
                                duration = end - start
                                if duration > 0:
                                    seg["duration"] = duration
                                    word_count = seg.get("word_count", word_count)
                                    seg["wpm"] = (word_count / duration) * 60
                                    seg["wps"] = word_count / duration
                            
                        self.all_speech_rate_results.extend(result)
                        for seg in result:
                            comparison = seg.get("comparison")
                            wpm = seg.get("wpm", 0)
                            if comparison == "too_fast":
                                msg = f"발화 속도가 빠릅니다: {wpm:.0f} WPM"
                                print(f"🔔 {msg}")
                                self.feedback_queue.put(msg)
                                self.ui_feedback_queue.put({"message": msg, "type": "speech_rate"})
                            elif comparison == "too_slow":
                                msg = f"발화 속도가 느립니다: {wpm:.0f} WPM"
                                print(f"🔔 {msg}")
                                self.feedback_queue.put(msg)
                                self.ui_feedback_queue.put({"message": msg, "type": "speech_rate"})
                
                except Exception as e:
                    print(f"⚠️ Error in {task_name} analysis: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Slow analyses (grammar, context) - run periodically but more frequently
        # Reduced from 10 seconds to 5 seconds for faster feedback
        current_time = time.time()
        if current_time - self.last_analysis_time > 5.0:
            # Run synchronously to ensure completion before session ends
            try:
                await self._run_slow_analysis()
            except Exception as e:
                print(f"⚠️ Slow analysis error: {e}")
                import traceback
                traceback.print_exc()
            self.last_analysis_time = current_time
    
    async def _run_slow_analysis(self):
        """Run slow analyses (grammar, context)."""
        if not self.transcript_buffer:
            return
        
        # Convert buffer to segment format with estimated durations
        segments = []
        for item in self.transcript_buffer:
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
                    if grammar_errors:
                        msg = f"문법 오류 {len(grammar_errors)}개 발견"
                        print(f"🔔 {msg}")
                        for i, error in enumerate(grammar_errors, 1):
                            details = error.get("error_details", {})
                            print(f"   {i}. '{details.get('original')}' → '{details.get('corrected')}'")
                        self.feedback_queue.put(msg)
                        self.ui_feedback_queue.put({"message": msg, "type": "grammar"})
                
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
                    if context_errors:
                        msg = f"맥락 오류 {len(context_errors)}개 발견"
                        print(f"🔔 {msg}")
                        for i, error in enumerate(context_errors, 1):
                            print(f"   {i}. '{error.get('utterance')}' - {error.get('reasoning')}")
                        self.feedback_queue.put(msg)
                        self.ui_feedback_queue.put({"message": msg, "type": "context"})
            
            except Exception as e:
                print(f"⚠️ Error in LLM analysis: {e}")
    
    def audio_callback(self, audio_chunk: bytes):
        """
        Callback to receive audio chunks from the streaming STT.
        
        Args:
            audio_chunk: Raw audio bytes
        """
        try:
            # Send to UI queue for visualization
            self.audio_queue.put(audio_chunk)
            
            # Add to buffer for stutter analysis
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_buffer.extend(audio_array)
            
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
    
    def streaming_producer(self):
        """Streaming producer: captures audio and sends to GCP STT."""
        print("🎤 Streaming producer started")
        
        try:
            # Initialize streaming STT with callbacks
            self.streaming_stt = GoogleSTTStreaming(
                callback=self.stt_callback,
                audio_callback=self.audio_callback
            )
            
            # Start streaming (blocking, includes audio capture)
            self.streaming_stt.start_streaming()
            
        except Exception as e:
            print(f"❌ Error in streaming producer: {e}")
            traceback.print_exc()
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
            traceback.print_exc()
            self.stop_event.set()
    
    def generate_summary_report(self):
        """Generate and print a comprehensive summary report after the session ends."""
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
                    print(f"\n  {i}. [{error.get('speaker')}] \"{error.get('utterance')}\"")
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
                    import traceback
                    traceback.print_exc()
            else:
                print("분석할 오디오 데이터가 없습니다.")
        
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
                "custom_keywords": self.custom_keywords,
                "target_wpm": self.target_wpm
            }
            
            pdf_path = self.report_generator.generate_report(session_data)
            print(f"✅ PDF 리포트 생성 완료: {pdf_path}")
        except Exception as e:
            print(f"❌ PDF 리포트 생성 실패: {e}")
            traceback.print_exc()
    
    def run(self, enable_ui: bool = True):
        """Run the main HabitLink session."""
        # Initialize components
        if not self.initialize_components():
            print("❌ 초기화 실패. 프로그램을 종료합니다.")
            return
        
        # Select analyses
        self.select_analyses()
        
        # Check if at least one analysis is enabled
        if not any(self.enabled_analyses.values()):
            print("\n⚠️ 활성화된 분석이 없습니다. 프로그램을 종료합니다.")
            return
        
        # Prepare session
        self.prepare_session()
        
        # Record session start time
        self.session_start_time = datetime.now()
        print(f"\n🕐 세션 시작 시각: {self.session_start_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}")
        
        # Start streaming thread
        streaming_thread = threading.Thread(target=self.streaming_producer, daemon=True)
        streaming_thread.start()
        
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
        
        # Wait for threads to finish
        print("마지막 분석을 완료하는 중...")
        streaming_thread.join(timeout=5)
        console_thread.join(timeout=2)
        
        # Generate summary report
        self.generate_summary_report()
