#!/usr/bin/env python3
"""
HabitLink Main Application
Multi-threaded Producer-Consumer architecture for real-time speech analysis
"""

import os
import sys
import time
import asyncio
import logging
import traceback
import threading
from queue import Queue, Empty
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import concurrent.futures

from src.audio_engine import AudioEngine
from src.stt import GoogleSTTDiarizer
from src.word_analyzer import WordAnalyzer
from src.speech_rate_analyzer import SpeechRateAnalyzer
from src.text_analyzer import TextAnalyzer
from src.utils import load_profanity_list

# --- Suppress library warnings ---
logging.getLogger('pyannote').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
os.environ["PYTORCH_SUPPRESS_DEPRECATION_WARNINGS"] = "1"
os.environ["PL_SUPPRESS_FORK"] = "1"


class HabitLinkSession:
    """
    Main class for managing a HabitLink analysis session with multi-threaded architecture.
    """
    
    def __init__(self):
        """Initialize the HabitLink session with default settings."""
        self.audio_engine = None
        self.diarizer = None
        self.word_analyzer = None
        self.speech_rate_analyzer = None
        self.text_analyzer = None
        self.profanity_list = []
        
        # User configuration
        self.enabled_analyses = {
            "keyword_detection": False,
            "profanity_detection": False,
            "speech_rate": False,
            "grammar": False,
            "context": False
        }
        self.custom_keywords = []
        self.target_wpm = None
        
        # Threading components
        self.task_queue = Queue()
        self.feedback_queue = Queue()
        self.stop_event = threading.Event()
        self.results_store = []
        
        # Audio recording settings
        self.chunk_duration = 10.0  # 10 seconds per chunk
        
    def initialize_components(self):
        """Initialize all analysis components."""
        print("\n🚀 Initializing HabitLink components...")
        
        try:
            # Initialize audio engine
            self.audio_engine = AudioEngine(samplerate=16000, channels=1)
            print("✅ Audio engine initialized")
            
            # Initialize STT with Google Cloud
            self.diarizer = GoogleSTTDiarizer()
            print("✅ Google Cloud STT initialized")
            
            # Initialize analyzers
            self.word_analyzer = WordAnalyzer()
            self.speech_rate_analyzer = SpeechRateAnalyzer()
            self.text_analyzer = TextAnalyzer()
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
            calibration_text = "죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다."
            print(f"\n\"{calibration_text}\"\n")
            input("준비가 되셨으면 Enter 키를 누르고 위 문장을 읽기 시작하세요...")
            
            try:
                # Record calibration audio
                calibration_duration = 15.0  # 15 seconds for calibration
                calibration_path = self.audio_engine.record(calibration_duration, "calibration_temp.wav")
                
                # Process with STT
                print("발화 속도를 분석 중...")
                calibration_transcript = self.diarizer.process(calibration_path)
                
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
    
    def audio_producer(self):
        """Audio producer thread: records audio chunks and adds them to the queue."""
        print("🎤 Audio producer started. Recording...")
        chunk_counter = 0
        
        while not self.stop_event.is_set():
            try:
                chunk_counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = f"temp_chunk_{timestamp}_{chunk_counter}.wav"
                
                # Record audio chunk
                self.audio_engine.record(self.chunk_duration, audio_path)
                
                # Add to processing queue
                self.task_queue.put({
                    "audio_path": audio_path,
                    "chunk_id": chunk_counter,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"❌ Error in audio producer: {e}")
        
        print("🎤 Audio producer stopping.")
    
    def analysis_consumer(self):
        """Analysis consumer thread: processes audio from the queue."""
        print("📊 Analysis consumer started.")
        
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.stop_event.is_set() or not self.task_queue.empty():
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1)
                audio_path = task["audio_path"]
                chunk_id = task["chunk_id"]
                
                # Process the audio chunk
                results = loop.run_until_complete(self._process_audio_chunk(audio_path, chunk_id))
                
                # Store results
                if results:
                    self.results_store.append(results)
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
                self.task_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"❌ Error in analysis consumer: {e}")
                    traceback.print_exc()
        
        loop.close()
        print("📊 Analysis consumer stopping.")
    
    async def _process_audio_chunk(self, audio_path: str, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Process a single audio chunk through the analysis pipeline."""
        try:
            # Step 1: Diarization
            diarized_transcript = self.diarizer.process(audio_path)
            
            if not diarized_transcript:
                return None
            
            # Step 2: Run enabled analyses
            results = {
                "chunk_id": chunk_id,
                "timestamp": time.time(),
                "transcript": diarized_transcript,
                "detected_keywords": [],
                "detected_profanity": [],
                "speech_rate_analysis": [],
                "grammar_analysis": [],
                "context_analysis": []
            }
            
            # Prepare concurrent tasks
            loop = asyncio.get_running_loop()
            tasks = []
            
            with concurrent.futures.ThreadPoolExecutor() as pool:
                # Keyword detection (fast, local)
                if self.enabled_analyses["keyword_detection"] and self.custom_keywords:
                    keyword_task = loop.run_in_executor(
                        pool, self.word_analyzer.analyze, diarized_transcript, self.custom_keywords
                    )
                    tasks.append(("keywords", keyword_task))
                
                # Profanity detection (fast, local)
                if self.enabled_analyses["profanity_detection"]:
                    profanity_task = loop.run_in_executor(
                        pool, self.word_analyzer.analyze, diarized_transcript, self.profanity_list
                    )
                    tasks.append(("profanity", profanity_task))
                
                # Speech rate analysis (fast, local)
                if self.enabled_analyses["speech_rate"]:
                    speech_rate_task = loop.run_in_executor(
                        pool, self.speech_rate_analyzer.analyze, diarized_transcript
                    )
                    tasks.append(("speech_rate", speech_rate_task))
                
                # LLM-based analysis (slow, network-bound) - runs asynchronously
                if self.enabled_analyses["grammar"] or self.enabled_analyses["context"]:
                    llm_task = self.text_analyzer.analyze(diarized_transcript)
                    tasks.append(("llm", llm_task))
                
                # Wait for all tasks to complete
                for task_name, task in tasks:
                    try:
                        result = await task
                        
                        if task_name == "keywords":
                            results["detected_keywords"] = result
                            # Real-time feedback for keywords
                            self._send_keyword_feedback(result, chunk_id)
                        
                        elif task_name == "profanity":
                            results["detected_profanity"] = result
                            # Real-time feedback for profanity
                            self._send_profanity_feedback(result, chunk_id)
                        
                        elif task_name == "speech_rate":
                            results["speech_rate_analysis"] = result
                            # Real-time feedback for speech rate
                            self._send_speech_rate_feedback(result, chunk_id)
                        
                        elif task_name == "llm":
                            results["grammar_analysis"] = result.get("grammar_errors", [])
                            results["context_analysis"] = result.get("context_errors", [])
                            # LLM results are primarily for the final report
                    
                    except Exception as e:
                        print(f"⚠️ Error in {task_name} analysis: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing audio chunk {chunk_id}: {e}")
            return None
    
    def _send_keyword_feedback(self, detected_keywords: List[Dict], chunk_id: int):
        """Send real-time feedback for detected keywords."""
        if detected_keywords:
            for item in detected_keywords:
                keyword = item.get("keyword")
                timestamp = item.get("timestamp", 0)
                speaker = item.get("speaker", "UNKNOWN")
                self.feedback_queue.put(
                    f"[청크 {chunk_id}] 🔍 키워드 검출: '{keyword}' ({speaker}, {timestamp:.2f}s)"
                )
    
    def _send_profanity_feedback(self, detected_profanity: List[Dict], chunk_id: int):
        """Send real-time feedback for detected profanity."""
        if detected_profanity:
            for item in detected_profanity:
                profanity = item.get("keyword")
                timestamp = item.get("timestamp", 0)
                speaker = item.get("speaker", "UNKNOWN")
                self.feedback_queue.put(
                    f"[청크 {chunk_id}] ⚠️ 비속어 검출: '{profanity}' ({speaker}, {timestamp:.2f}s)"
                )
    
    def _send_speech_rate_feedback(self, speech_rate_analysis: List[Dict], chunk_id: int):
        """Send real-time feedback for speech rate analysis."""
        if speech_rate_analysis and self.target_wpm:
            for segment in speech_rate_analysis:
                comparison = segment.get("comparison")
                if comparison == "too_fast":
                    wpm = segment.get("wpm", 0)
                    speaker = segment.get("speaker", "UNKNOWN")
                    self.feedback_queue.put(
                        f"[청크 {chunk_id}] 🏃 발화 속도가 빠릅니다: {wpm:.0f} WPM ({speaker})"
                    )
                elif comparison == "too_slow":
                    wpm = segment.get("wpm", 0)
                    speaker = segment.get("speaker", "UNKNOWN")
                    self.feedback_queue.put(
                        f"[청크 {chunk_id}] 🐢 발화 속도가 느립니다: {wpm:.0f} WPM ({speaker})"
                    )
    
    def feedback_loop(self):
        """Main thread feedback loop: prints real-time feedback to the console."""
        print("\n" + "="*60)
        print("🎙️ 세션 시작")
        print("="*60)
        print("실시간 피드백이 아래에 표시됩니다.")
        print("세션을 종료하려면 Ctrl+C를 누르세요.")
        print("="*60 + "\n")
        
        while not self.stop_event.is_set():
            try:
                # Get feedback from queue with timeout
                feedback = self.feedback_queue.get(timeout=0.1)
                print(feedback)
            except Empty:
                continue
    
    def generate_summary_report(self):
        """Generate and print a comprehensive summary report after the session ends."""
        print("\n\n" + "="*60)
        print("📋 세션 요약 리포트")
        print("="*60)
        
        if not self.results_store:
            print("\n분석할 데이터가 없습니다.")
            return
        
        # Aggregate all transcripts
        print("\n--- ✅ 전체 대화 내용 ---")
        for result in self.results_store:
            chunk_id = result.get("chunk_id")
            transcript = result.get("transcript", [])
            if transcript:
                print(f"\n[청크 {chunk_id}]")
                for segment in transcript:
                    speaker = segment.get("speaker", "UNKNOWN")
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "")
                    print(f"  [{start:.2f}s - {end:.2f}s] {speaker}: {text}")
        
        # Keyword detection summary
        if self.enabled_analyses["keyword_detection"]:
            print("\n--- 🔍 키워드 검출 요약 ---")
            all_keywords = []
            for result in self.results_store:
                all_keywords.extend(result.get("detected_keywords", []))
            
            if all_keywords:
                keyword_counts = defaultdict(int)
                for item in all_keywords:
                    keyword_counts[item["keyword"].lower()] += 1
                
                print(f"총 {len(all_keywords)}회 검출:")
                for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - '{keyword}': {count}회")
            else:
                print("검출된 키워드가 없습니다.")
        
        # Profanity detection summary
        if self.enabled_analyses["profanity_detection"]:
            print("\n--- ⚠️ 비속어 검출 요약 ---")
            all_profanity = []
            for result in self.results_store:
                all_profanity.extend(result.get("detected_profanity", []))
            
            if all_profanity:
                profanity_counts = defaultdict(int)
                for item in all_profanity:
                    profanity_counts[item["keyword"].lower()] += 1
                
                print(f"총 {len(all_profanity)}회 검출:")
                for profanity, count in sorted(profanity_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - '{profanity}': {count}회")
            else:
                print("검출된 비속어가 없습니다.")
        
        # Speech rate analysis summary
        if self.enabled_analyses["speech_rate"]:
            print("\n--- 🏃 발화 속도 분석 요약 ---")
            all_speech_rate = []
            for result in self.results_store:
                all_speech_rate.extend(result.get("speech_rate_analysis", []))
            
            if all_speech_rate:
                total_word_count = sum(seg.get("word_count", 0) for seg in all_speech_rate)
                total_duration = sum(seg.get("duration", 0) for seg in all_speech_rate)
                
                if total_duration > 0:
                    overall_wpm = (total_word_count / total_duration) * 60
                    print(f"전체 평균 발화 속도: {overall_wpm:.2f} WPM")
                    
                    if self.target_wpm:
                        print(f"목표 발화 속도: {self.target_wpm:.2f} WPM")
                        
                        # Count too_fast and too_slow instances
                        too_fast = sum(1 for seg in all_speech_rate if seg.get("comparison") == "too_fast")
                        too_slow = sum(1 for seg in all_speech_rate if seg.get("comparison") == "too_slow")
                        good = sum(1 for seg in all_speech_rate if seg.get("comparison") == "good")
                        
                        print(f"\n발화 속도 분포:")
                        print(f"  - 적절: {good}회")
                        print(f"  - 너무 빠름: {too_fast}회")
                        print(f"  - 너무 느림: {too_slow}회")
                    
                    # Speaker breakdown
                    speaker_stats = defaultdict(lambda: {"word_count": 0, "duration": 0})
                    for seg in all_speech_rate:
                        speaker = seg.get("speaker", "UNKNOWN")
                        speaker_stats[speaker]["word_count"] += seg.get("word_count", 0)
                        speaker_stats[speaker]["duration"] += seg.get("duration", 0)
                    
                    print("\n화자별 발화 속도:")
                    for speaker, stats in speaker_stats.items():
                        if stats["duration"] > 0:
                            speaker_wpm = (stats["word_count"] / stats["duration"]) * 60
                            print(f"  - {speaker}: {speaker_wpm:.2f} WPM")
            else:
                print("발화 속도 분석 결과가 없습니다.")
        
        # Grammar analysis summary
        if self.enabled_analyses["grammar"]:
            print("\n--- 🧐 문법 분석 요약 ---")
            all_grammar_errors = []
            for result in self.results_store:
                all_grammar_errors.extend(result.get("grammar_analysis", []))
            
            if all_grammar_errors:
                print(f"총 {len(all_grammar_errors)}개의 문법 오류 발견:")
                for i, error in enumerate(all_grammar_errors[:10], 1):  # Show first 10
                    details = error.get("error_details", {})
                    print(f"\n  {i}. [{error.get('speaker')}] '{details.get('original')}' → '{details.get('corrected')}'")
                    print(f"     설명: {details.get('explanation')}")
                
                if len(all_grammar_errors) > 10:
                    print(f"\n  ... 그 외 {len(all_grammar_errors) - 10}개 더")
            else:
                print("문법 오류가 발견되지 않았습니다.")
        
        # Context analysis summary
        if self.enabled_analyses["context"]:
            print("\n--- 🧠 맥락 분석 요약 ---")
            all_context_errors = []
            for result in self.results_store:
                all_context_errors.extend(result.get("context_analysis", []))
            
            if all_context_errors:
                print(f"총 {len(all_context_errors)}개의 맥락 오류 발견:")
                for i, error in enumerate(all_context_errors[:5], 1):  # Show first 5
                    print(f"\n  {i}. [{error.get('speaker')}] \"{error.get('utterance')}\"")
                    print(f"     분석: {error.get('reasoning')}")
                
                if len(all_context_errors) > 5:
                    print(f"\n  ... 그 외 {len(all_context_errors) - 5}개 더")
            else:
                print("맥락 오류가 발견되지 않았습니다.")
        
        print("\n" + "="*60)
        print("세션 종료")
        print("="*60)
    
    def run(self):
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
        
        # Start threads
        producer_thread = threading.Thread(target=self.audio_producer, daemon=True)
        consumer_thread = threading.Thread(target=self.analysis_consumer, daemon=True)
        
        producer_thread.start()
        consumer_thread.start()
        
        try:
            # Main feedback loop
            self.feedback_loop()
        except KeyboardInterrupt:
            print("\n\n⏸️ 세션을 종료하는 중...")
            self.stop_event.set()
        
        # Wait for threads to finish
        print("마지막 분석을 완료하는 중...")
        producer_thread.join(timeout=2)
        consumer_thread.join(timeout=10)
        
        # Generate summary report
        self.generate_summary_report()


def main():
    """Main entry point for the HabitLink application."""
    load_dotenv()
    
    # Pre-flight checks
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        print("Please set your GROQ_API_KEY to use LLM-based analysis features.")
        return
    
    # Check Google Cloud credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.path.exists("gcp_credentials.json"):
        print("❌ Error: Google Cloud credentials not found.")
        print("Please either:")
        print("  1. Set GOOGLE_APPLICATION_CREDENTIALS in your .env file, or")
        print("  2. Run 'gcloud auth application-default login'")
        return
    
    print("="*60)
    print("🎯 HabitLink: AI-Powered Korean Speech Habit Correction System")
    print("="*60)
    print("\n실시간 음성 분석 세션을 시작합니다.")
    
    # Create and run session
    session = HabitLinkSession()
    session.run()
    
    print("\n\n감사합니다. HabitLink를 사용해 주셔서 감사합니다! 👋")


if __name__ == "__main__":
    main()

