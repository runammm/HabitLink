import numpy as np
import librosa
from collections import deque
from typing import List, Dict, Any, Optional
import time


class StutterDetector:
    """
    Detects stuttering events from raw audio stream in real-time.
    Uses audio signal processing to identify repetitions, prolongations, and blocks
    WITHOUT relying on text from STT (which may be pre-cleaned).
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 energy_threshold: float = 0.025, 
                 repetition_similarity_threshold: float = 0.89, 
                 prolongation_duration_threshold: float = 0.95, 
                 silence_duration_threshold: float = 1.2): 
        """
        Initialize the real-time stutter detector.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            energy_threshold: Energy threshold to distinguish speech from silence
            repetition_similarity_threshold: Correlation threshold for repetition detection
            prolongation_duration_threshold: Duration (seconds) to flag prolongations
            silence_duration_threshold: Duration (seconds) to flag blocks
        """
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.repetition_threshold = repetition_similarity_threshold
        self.prolongation_threshold = prolongation_duration_threshold
        self.silence_threshold = silence_duration_threshold
        
        # Sliding window buffer (last 3 seconds of audio)
        self.audio_buffer = deque(maxlen=sample_rate * 3)
        
        # Detection results
        self.detected_events = []
        self.last_analysis_time = time.time()
        
        # State tracking
        self.in_speech = False
        self.speech_start_time = None
        self.last_segment_features = None
    
    def add_audio_chunk(self, audio_chunk: bytes):
        """
        Add audio chunk to the buffer for real-time analysis.
        
        Args:
            audio_chunk: Raw audio bytes (int16)
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.extend(audio_array)
            
            # Analyze every 0.7 seconds (중간값: 0.5와 1.0의 중간보다 약간 길게)
            current_time = time.time()
            if current_time - self.last_analysis_time >= 0.7:
                self._analyze_current_buffer(current_time)
                self.last_analysis_time = current_time
        
        except Exception as e:
            pass  # Silently ignore errors in real-time processing
    
    def _analyze_current_buffer(self, current_time: float):
        """Analyze the current audio buffer for stuttering patterns."""
        if len(self.audio_buffer) < self.sr * 0.5:  # Need at least 0.5 seconds
            return
        
        # Convert buffer to numpy array
        audio = np.array(self.audio_buffer)
        
        # Check for repetitions
        self._detect_realtime_repetitions(audio, current_time)
        
        # Check for prolongations
        self._detect_realtime_prolongations(audio, current_time)
        
        # Check for blocks (sudden silences within speech)
        self._detect_realtime_blocks(audio, current_time)
    
    def _detect_realtime_repetitions(self, audio: np.ndarray, timestamp: float):
        """
        Detect repetitions by comparing recent audio segments.
        Uses cross-correlation to find similar consecutive segments.
        """
        try:
            # Split audio into short segments (200ms each)
            segment_length = int(self.sr * 0.2)  # 200ms
            
            if len(audio) < segment_length * 2:
                return
            
            # Get last few segments
            num_segments = min(5, len(audio) // segment_length)
            segments = []
            
            for i in range(num_segments):
                start = len(audio) - (i + 1) * segment_length
                end = len(audio) - i * segment_length
                if start >= 0:
                    segment = audio[start:end]
                    
                    # Extract MFCC features
                    if len(segment) > 0 and np.max(np.abs(segment)) > self.energy_threshold:
                        mfcc = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13)
                        segments.append({
                            'mfcc': mfcc,
                            'audio': segment,
                            'energy': np.mean(segment ** 2)
                        })
            
            # Compare consecutive segments for similarity
            for i in range(len(segments) - 1):
                seg1 = segments[i]
                seg2 = segments[i + 1]
                
                # Only compare if both have speech energy
                if seg1['energy'] > self.energy_threshold and seg2['energy'] > self.energy_threshold:
                    # Calculate correlation between MFCC features
                    mfcc1_mean = np.mean(seg1['mfcc'], axis=1)
                    mfcc2_mean = np.mean(seg2['mfcc'], axis=1)
                    
                    # Normalize and compute correlation
                    mfcc1_norm = (mfcc1_mean - np.mean(mfcc1_mean)) / (np.std(mfcc1_mean) + 1e-8)
                    mfcc2_norm = (mfcc2_mean - np.mean(mfcc2_mean)) / (np.std(mfcc2_mean) + 1e-8)
                    
                    correlation = np.corrcoef(mfcc1_norm, mfcc2_norm)[0, 1]
                    
                    # If highly correlated, it's likely a repetition
                    if correlation > self.repetition_threshold:
                        event = {
                            'type': 'repetition',
                            'timestamp': timestamp - (i * 0.2),  # Approximate timestamp
                            'confidence': float(correlation),
                            'severity': 'moderate' if correlation > 0.9 else 'mild'
                        }
                        
                        # Avoid duplicate detections
                        if not self._is_duplicate_event(event):
                            self.detected_events.append(event)
        
        except Exception as e:
            pass  # Silently handle errors
    
    def _detect_realtime_prolongations(self, audio: np.ndarray, timestamp: float):
        """
        Detect prolongations by finding sustained single-frequency sounds.
        """
        try:
            # Calculate zero-crossing rate (lower for sustained sounds)
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Find regions with low ZCR and sustained energy (prolongations)
            prolonged_frames = 0
            for i in range(len(zcr)):
                # 중간 조건: ZCR < 0.09, RMS > threshold * 1.3
                if zcr[i] < 0.09 and rms[i] > self.energy_threshold * 1.3:
                    prolonged_frames += 1
                else:
                    if prolonged_frames > 0:
                        duration = prolonged_frames * self.hop_length / self.sr
                        
                        if duration > self.prolongation_threshold:
                            event = {
                                'type': 'prolongation',
                                'timestamp': timestamp - duration,
                                'duration': round(duration, 2),
                                'severity': 'severe' if duration > 1.4 else 'moderate'  # 중간값
                            }
                            
                            if not self._is_duplicate_event(event):
                                self.detected_events.append(event)
                        
                        prolonged_frames = 0
        
        except Exception as e:
            pass
    
    def _detect_realtime_blocks(self, audio: np.ndarray, timestamp: float):
        """
        Detect blocks (sudden silences within continuous speech).
        """
        try:
            # Split into speech/silence using energy threshold
            intervals = librosa.effects.split(audio, top_db=30, frame_length=self.frame_length, hop_length=self.hop_length)
            
            # If we have multiple intervals, check for gaps (blocks)
            if len(intervals) > 1:
                for i in range(len(intervals) - 1):
                    gap_start = intervals[i][1]
                    gap_end = intervals[i + 1][0]
                    gap_duration = (gap_end - gap_start) / self.sr
                    
                    # Block 검출: 1.2초 이상 2.5초 미만 (중간 범위)
                    if 1.2 < gap_duration < 2.5:
                        event = {
                            'type': 'block',
                            'timestamp': timestamp - ((len(audio) - gap_start) / self.sr),
                            'duration': round(gap_duration, 2),
                            'severity': 'severe' if gap_duration > 1.5 else 'moderate'
                        }
                        
                        if not self._is_duplicate_event(event):
                            self.detected_events.append(event)
        
        except Exception as e:
            pass
    
    def _is_duplicate_event(self, new_event: Dict[str, Any], time_window: float = 1.8) -> bool:
        """
        Check if this event is a duplicate of a recent event.
        
        Args:
            new_event: The event to check
            time_window: Time window (seconds) to consider for duplicates (중간값: 1.0과 2.5의 중간보다 약간 높게)
        
        Returns:
            True if duplicate, False otherwise
        """
        new_timestamp = new_event['timestamp']
        new_type = new_event['type']
        
        for event in self.detected_events[-10:]:  # Check last 10 events
            if event['type'] == new_type:
                if abs(event['timestamp'] - new_timestamp) < time_window:
                    return True
        
        return False
    
    def get_detected_events(self) -> List[Dict[str, Any]]:
        """
        Get all detected stuttering events.
        
        Returns:
            List of detected events
        """
        return self.detected_events.copy()
    
    def get_recent_events(self, time_window: float = 10.0) -> List[Dict[str, Any]]:
        """
        Get recent stuttering events within a time window.
        
        Args:
            time_window: Time window (seconds) to look back
        
        Returns:
            List of recent events
        """
        current_time = time.time()
        recent_events = [
            event for event in self.detected_events
            if current_time - event['timestamp'] < time_window
        ]
        return recent_events
    
    def clear_events(self):
        """Clear all detected events."""
        self.detected_events.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected stuttering events.
        
        Returns:
            Dictionary with statistics
        """
        if not self.detected_events:
            return {
                'total_events': 0,
                'repetitions': 0,
                'prolongations': 0,
                'blocks': 0
            }
        
        type_counts = {
            'repetition': 0,
            'prolongation': 0,
            'block': 0
        }
        
        for event in self.detected_events:
            event_type = event['type']
            if event_type in type_counts:
                type_counts[event_type] += 1
        
        return {
            'total_events': len(self.detected_events),
            'repetitions': type_counts['repetition'],
            'prolongations': type_counts['prolongation'],
            'blocks': type_counts['block']
        }

