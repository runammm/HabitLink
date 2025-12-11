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
    
    Based on research-validated criteria:
    1. Blocks (막힘):
       - Intra-lexical pauses (단어-내 멈춤): ≥150ms silence within words
       - Inter-lexical pauses (단어-간 멈춤): ≥250ms are normal hesitations
       Reference: Research shows normal speakers rarely have >200ms pauses within words
    
    2. Prolongations (연장):
       - Absolute: Single phoneme sustained ≥800ms (relaxed threshold)
       - Relative: ≥(mean + 2.5×SD) of user's average phoneme duration
       Reference: 800ms is clearly perceived as "abnormally long" by listeners
       Note: 250ms too sensitive; adjusted to 800ms for real-world use
    
    3. Repetitions (반복):
       - MFCC correlation coefficient >0.92 between consecutive frames (20-40ms)
       - 2+ occurrences of similar patterns
       Reference: MFCC correlation >0.92 identifies repetition segments
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 energy_threshold: float = 0.02,  # Energy threshold for VAD
                 # Repetition: MFCC correlation >0.96 (slightly relaxed from 0.98)
                 repetition_similarity_threshold: float = 0.96,
                 # Prolongation: 800ms absolute threshold (stricter - research-based)
                 prolongation_duration_threshold: float = 0.8,
                 # Block (intra-lexical): 130ms within words (slightly relaxed from 150ms)
                 intra_lexical_silence_threshold: float = 0.13,
                # Hesitation (inter-lexical): 250ms between words (normal)
                inter_lexical_silence_threshold: float = 0.25,
                # Long pause (sentence-ending): 1.5s or more (natural sentence boundary)
                sentence_pause_threshold: float = 1.5,
                # Noise reduction settings
                enable_noise_reduction: bool = True):
        """
        Initialize the real-time stutter detector with research-based thresholds.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_length: Frame length for analysis (default: 2048 samples ≈ 128ms)
            hop_length: Hop length for analysis (default: 512 samples ≈ 32ms)
            energy_threshold: Energy threshold for VAD (Voice Activity Detection)
            repetition_similarity_threshold: MFCC correlation for repetition (≥0.96, strict)
            prolongation_duration_threshold: Minimum duration for prolongation (800ms, stricter)
            intra_lexical_silence_threshold: Silence threshold within words (130ms, slightly relaxed)
            inter_lexical_silence_threshold: Silence threshold between words (250ms, normal)
            sentence_pause_threshold: Silence threshold for sentence boundaries (1.5s, normal)
            enable_noise_reduction: Enable noise reduction preprocessing
        """
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.repetition_threshold = repetition_similarity_threshold
        self.prolongation_threshold = prolongation_duration_threshold
        self.intra_lexical_threshold = intra_lexical_silence_threshold
        self.inter_lexical_threshold = inter_lexical_silence_threshold
        self.sentence_pause_threshold = sentence_pause_threshold  # 문장 간 긴 쉼
        self.enable_noise_reduction = enable_noise_reduction
        
        # Sliding window buffer (last 3 seconds of audio)
        self.audio_buffer = deque(maxlen=sample_rate * 3)
        
        # Detection results
        self.detected_events = []
        self.last_analysis_time = time.time()
        
        # State tracking for context-aware detection
        self.recent_speech_segments = deque(maxlen=10)  # Track recent speech segments
        self.in_speech = False
        self.speech_start_time = None
        self.last_speech_end_time = None
    
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
            
            # Analyze every 0.5 seconds for balanced real-time detection
            current_time = time.time()
            if current_time - self.last_analysis_time >= 0.5:
                self._analyze_current_buffer(current_time)
                self.last_analysis_time = current_time
        
        except Exception as e:
            pass  # Silently ignore errors in real-time processing
    
    def _analyze_current_buffer(self, current_time: float):
        """Analyze the current audio buffer for stuttering patterns."""
        if len(self.audio_buffer) < self.sr * 0.3:  # Need at least 300ms
            return
        
        # Convert buffer to numpy array
        audio = np.array(self.audio_buffer)
        
        # Apply noise reduction if enabled
        if self.enable_noise_reduction:
            try:
                from .audio_utils import reduce_noise
                # Convert to float32 for noise reduction
                audio_float = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
                audio_float = reduce_noise(audio_float, self.sr, stationary=True, prop_decrease=0.8)
                # Convert back to original scale
                audio = audio_float
            except Exception as e:
                pass  # Continue with original audio if noise reduction fails
        
        # Detect repetitions using MFCC correlation
        self._detect_repetitions_mfcc(audio, current_time)
        
        # Detect prolongations using spectral stability
        self._detect_prolongations(audio, current_time)
        
        # Detect blocks (intra-lexical pauses) using contextual VAD
        self._detect_blocks_contextual(audio, current_time)
    
    def _detect_repetitions_mfcc(self, audio: np.ndarray, timestamp: float):
        """
        Detect repetitions using Onset Detection + Segment Comparison.
        
        New approach (redesigned):
        1. Detect syllable onsets (start points of each sound unit)
        2. Extract segments between onsets
        3. Compare adjacent segments using MFCC similarity
        4. If similar segments repeat consecutively, it's a repetition
        
        This catches fast repetitions like "이이이거", "봐봐봐라" that the
        old consecutive-frame approach couldn't detect.
        """
        try:
            # Need at least 200ms of audio
            if len(audio) < self.sr * 0.2:
                return
            
            # ===== Method 1: Onset-based Segment Comparison =====
            # Only use onset-based method (self_similarity causes too many false positives)
            self._detect_repetitions_onset_based(audio, timestamp)
            
            # ===== Method 2: Self-Similarity Matrix (DISABLED - too many false positives) =====
            # self._detect_repetitions_self_similarity(audio, timestamp)
            
        except Exception as e:
            pass
    
    def _detect_repetitions_onset_based(self, audio: np.ndarray, timestamp: float):
        """
        Onset-based repetition detection.
        Detects syllable boundaries and compares adjacent syllables.
        """
        try:
            # Detect onsets (syllable start points)
            # Use a sensitive onset detector for fast repetitions
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sr,
                units='frames',
                hop_length=256,  # Finer resolution
                backtrack=True,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.15,  # Less sensitive (stricter)
                wait=4  # Minimum 4 frames between onsets (~64ms at 256 hop)
            )
            
            # Convert frames to samples
            onset_samples = librosa.frames_to_samples(onset_frames, hop_length=256)
            
            # Need at least 3 onsets to detect repetition (2 segments to compare)
            if len(onset_samples) < 3:
                return
            
            # Extract MFCC for each segment between onsets
            segments_mfcc = []
            segment_times = []
            
            for i in range(len(onset_samples) - 1):
                start = onset_samples[i]
                end = onset_samples[i + 1]
                
                # Skip very short or very long segments
                segment_duration = (end - start) / self.sr
                if segment_duration < 0.03 or segment_duration > 0.5:
                    continue
                
                segment = audio[start:end]
                
                # Check if segment has enough energy (not silence)
                segment_energy = np.sqrt(np.mean(segment ** 2))
                if segment_energy < self.energy_threshold:
                    continue
                
                # Extract MFCC for this segment
                mfcc = librosa.feature.mfcc(
                    y=segment,
                    sr=self.sr,
                    n_mfcc=13,
                    n_fft=min(512, len(segment)),
                    hop_length=min(256, len(segment) // 2)
                )
                
                # Average MFCC across time to get a single feature vector per segment
                mfcc_mean = np.mean(mfcc, axis=1)
                segments_mfcc.append(mfcc_mean)
                segment_times.append(start / self.sr)
            
            # Need at least 2 segments to compare
            if len(segments_mfcc) < 2:
                return
            
            # Compare adjacent segments
            repetition_count = 0
            repetition_start_time = None
            
            for i in range(len(segments_mfcc) - 1):
                mfcc1 = segments_mfcc[i]
                mfcc2 = segments_mfcc[i + 1]
                
                # Normalize
                mfcc1_norm = (mfcc1 - np.mean(mfcc1)) / (np.std(mfcc1) + 1e-8)
                mfcc2_norm = (mfcc2 - np.mean(mfcc2)) / (np.std(mfcc2) + 1e-8)
                
                # Calculate similarity (correlation only - cosine similarity causes false positives)
                similarity = np.corrcoef(mfcc1_norm, mfcc2_norm)[0, 1]
                
                # Skip if similarity is NaN
                if np.isnan(similarity):
                    continue
                
                # Threshold for segment similarity (extremely strict)
                if similarity > self.repetition_threshold:
                    if repetition_count == 0:
                        repetition_start_time = segment_times[i]
                    repetition_count += 1
                else:
                    # End of repetition sequence
                    if repetition_count >= 4:  # Need at least 5 similar segments (extremely strict)
                        event = {
                            'type': 'repetition',
                            'timestamp': timestamp - (len(audio) / self.sr) + (repetition_start_time or 0),
                            'count': repetition_count + 1,  # +1 because count is comparisons, not segments
                            'confidence': float(similarity),
                            'method': 'onset_based',
                            'severity': 'severe' if repetition_count >= 6 else 'moderate'
                        }
                        
                        if not self._is_duplicate_event(event):
                            self.detected_events.append(event)
                    
                    repetition_count = 0
                    repetition_start_time = None
            
            # Check for repetition at end of buffer
            if repetition_count >= 4:
                event = {
                    'type': 'repetition',
                    'timestamp': timestamp - (len(audio) / self.sr) + (repetition_start_time or 0),
                    'count': repetition_count + 1,
                    'confidence': 0.96,
                    'method': 'onset_based',
                    'severity': 'severe' if repetition_count >= 6 else 'moderate'
                }
                
                if not self._is_duplicate_event(event):
                    self.detected_events.append(event)
                    
        except Exception as e:
            pass
    
    def _detect_repetitions_self_similarity(self, audio: np.ndarray, timestamp: float):
        """
        Self-similarity matrix based repetition detection.
        Looks for diagonal patterns indicating repeated sounds.
        """
        try:
            # Extract MFCC with fine resolution
            hop_length = 256
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=13,
                n_fft=1024,
                hop_length=hop_length
            )
            
            # Calculate RMS for energy filtering
            rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length)[0]
            
            # Find frames with enough energy
            active_frames = np.where(rms > self.energy_threshold)[0]
            
            if len(active_frames) < 4:
                return
            
            # Build self-similarity matrix for active frames only
            # Look for repetitions with stride (comparing frame i with frame i+stride)
            min_stride = 3   # Minimum ~48ms apart (at 256 hop, 16kHz)
            max_stride = 15  # Maximum ~240ms apart
            
            for stride in range(min_stride, max_stride + 1):
                repetition_count = 0
                repetition_start_frame = None
                
                for i in range(len(active_frames) - stride):
                    frame_idx1 = active_frames[i]
                    frame_idx2 = active_frames[i] + stride
                    
                    if frame_idx2 >= mfcc.shape[1]:
                        break
                    
                    # Check if both frames have energy
                    if rms[frame_idx1] < self.energy_threshold or rms[frame_idx2] < self.energy_threshold:
                        continue
                    
                    mfcc1 = mfcc[:, frame_idx1]
                    mfcc2 = mfcc[:, frame_idx2]
                    
                    # Normalize
                    mfcc1_norm = (mfcc1 - np.mean(mfcc1)) / (np.std(mfcc1) + 1e-8)
                    mfcc2_norm = (mfcc2 - np.mean(mfcc2)) / (np.std(mfcc2) + 1e-8)
                    
                    # Calculate similarity
                    similarity = np.corrcoef(mfcc1_norm, mfcc2_norm)[0, 1]
                    
                    if not np.isnan(similarity) and similarity > self.repetition_threshold:
                        if repetition_count == 0:
                            repetition_start_frame = frame_idx1
                        repetition_count += 1
                    else:
                        if repetition_count >= 8:  # Need at least 8 matching pairs (very strict)
                            event = {
                                'type': 'repetition',
                                'timestamp': timestamp - (len(audio) / self.sr) + (repetition_start_frame * hop_length / self.sr),
                                'count': repetition_count,
                                'confidence': float(similarity) if not np.isnan(similarity) else 0.95,
                                'method': 'self_similarity',
                                'stride_ms': round(stride * hop_length / self.sr * 1000, 1),
                                'severity': 'severe' if repetition_count >= 12 else 'moderate'
                            }
                            
                            if not self._is_duplicate_event(event):
                                self.detected_events.append(event)
                        
                        repetition_count = 0
                        repetition_start_frame = None
                        
        except Exception as e:
            pass
    
    def _detect_prolongations(self, audio: np.ndarray, timestamp: float):
        """
        Detect prolongations using spectral stability.
        Research criterion: Single phoneme sustained ≥800ms (relaxed threshold).
        Note: 250ms too sensitive; 800ms captures true prolongations.
        """
        try:
            # Calculate spectral features
            zcr = librosa.feature.zero_crossing_rate(
                audio, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            )[0]
            
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            )[0]
            
            # Find sustained sounds (low ZCR + sustained energy)
            sustained_frames = 0
            
            for i in range(len(zcr)):
                # Low ZCR indicates sustained sound (vowels, fricatives)
                # Sufficient energy indicates active speech
                # ZCR threshold set to 0.12 (balanced)
                is_sustained = (zcr[i] < 0.06 and rms[i] > self.energy_threshold * 1.5)
                
                if is_sustained:
                    sustained_frames += 1
                else:
                    if sustained_frames > 0:
                        duration = sustained_frames * self.hop_length / self.sr
                        
                        # Relaxed criterion: ≥800ms (0.8s) to avoid false positives
                        if duration >= self.prolongation_threshold:
                            event = {
                                'type': 'prolongation',
                                'timestamp': timestamp - duration,
                                'duration': round(duration, 3),
                                'severity': 'severe' if duration >= 1.0 else 'moderate'
                            }
                            
                            if not self._is_duplicate_event(event):
                                self.detected_events.append(event)
                    
                    sustained_frames = 0
        
        except Exception as e:
            pass
    
    def _detect_blocks_contextual(self, audio: np.ndarray, timestamp: float):
        """
        Detect blocks (intra-lexical pauses) using contextual analysis.
        Key insight: Distinguish between different types of pauses:
        
        Research criteria:
        - Intra-lexical (within words): ≥150ms → Block (stuttering)
        - Inter-lexical (between words): 250ms-1.5s → Normal hesitation
        - Sentence boundary: ≥1.5s → Natural pause (NOT a block)
        """
        try:
            # Detect speech/silence intervals using energy-based VAD
            intervals = librosa.effects.split(
                audio, 
                top_db=30,  # Silence threshold
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            if len(intervals) < 2:
                return
            
            # Analyze gaps between speech intervals
            for i in range(len(intervals) - 1):
                speech1_end = intervals[i][1]
                speech2_start = intervals[i + 1][0]
                gap_duration = (speech2_start - speech1_end) / self.sr
                
                # Extract features of surrounding speech to determine context
                speech1_audio = audio[max(0, intervals[i][0]):speech1_end]
                speech2_audio = audio[speech2_start:min(len(audio), intervals[i+1][1])]
                
                # Calculate energy of surrounding speech
                speech1_energy = np.mean(speech1_audio ** 2) if len(speech1_audio) > 0 else 0
                speech2_energy = np.mean(speech2_audio ** 2) if len(speech2_audio) > 0 else 0
                
                # Skip very long pauses (likely sentence boundaries, not blocks)
                if gap_duration >= self.sentence_pause_threshold:
                    continue  # 1.5초 이상은 문장 간 쉼으로 간주, 정상
                
                # Both sides must have speech energy
                if speech1_energy > self.energy_threshold and speech2_energy > self.energy_threshold:
                    # Determine if this is likely intra-lexical or inter-lexical
                    # Heuristic: shorter speech segments with high energy continuity 
                    # are more likely to be within a word
                    speech1_duration = len(speech1_audio) / self.sr
                    speech2_duration = len(speech2_audio) / self.sr
                    
                    # If surrounding segments are short (<0.5s each), likely within a word
                    is_likely_intra_lexical = (speech1_duration < 0.5 and speech2_duration < 0.5)
                    
                    # Apply appropriate threshold
                    if is_likely_intra_lexical:
                        # Intra-lexical: ≥150ms is a block (research criterion)
                        if gap_duration >= self.intra_lexical_threshold:
                            event = {
                                'type': 'block',
                                'timestamp': timestamp - ((len(audio) - speech1_end) / self.sr),
                                'duration': round(gap_duration, 3),
                                'context': 'intra_lexical',
                                'severity': 'severe' if gap_duration >= 0.3 else 'moderate'
                            }
                            
                            if not self._is_duplicate_event(event):
                                self.detected_events.append(event)
                    else:
                        # Inter-lexical: ≥250ms might be normal hesitation
                        # Only flag if it's unusually long (>300ms), but not too long (sentence boundary)
                        # Relaxed from 500ms to 300ms
                        if gap_duration >= 0.45:
                            event = {
                                'type': 'block',
                                'timestamp': timestamp - ((len(audio) - speech1_end) / self.sr),
                                'duration': round(gap_duration, 3),
                                'context': 'inter_lexical',
                                'severity': 'moderate'
                            }
                            
                            if not self._is_duplicate_event(event):
                                self.detected_events.append(event)
        
        except Exception as e:
            pass
    
    def _is_duplicate_event(self, new_event: Dict[str, Any], time_window: float = 1.0) -> bool:
        """
        Check if this event is a duplicate of a recent event.
        
        Args:
            new_event: The event to check
            time_window: Time window (seconds) to consider for duplicates
        
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
                'blocks': 0,
                'intra_lexical_blocks': 0,
                'inter_lexical_blocks': 0
            }
        
        type_counts = {
            'repetition': 0,
            'prolongation': 0,
            'block': 0,
            'intra_lexical': 0,
            'inter_lexical': 0
        }
        
        for event in self.detected_events:
            event_type = event['type']
            if event_type in type_counts:
                type_counts[event_type] += 1
            
            # Count block contexts separately
            if event_type == 'block':
                context = event.get('context', 'unknown')
                if context in type_counts:
                    type_counts[context] += 1
        
        return {
            'total_events': len(self.detected_events),
            'repetitions': type_counts['repetition'],
            'prolongations': type_counts['prolongation'],
            'blocks': type_counts['block'],
            'intra_lexical_blocks': type_counts['intra_lexical'],
            'inter_lexical_blocks': type_counts['inter_lexical']
        }
