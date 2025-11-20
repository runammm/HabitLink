"""
Stutter Analyzer Module
Detects stuttering events using a hybrid approach (text + audio analysis).
"""

import re
import librosa
import numpy as np
from typing import List, Dict, Any


class StutterAnalyzer:
    """
    Analyzes audio and text to detect stuttering events like repetitions,
    prolongations, and blocks.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 prolongation_threshold_sec: float = 0.8,
                 block_threshold_sec: float = 1.0,
                 sentence_end_threshold_sec: float = 1.5,
                 silence_top_db: int = 30):
        """
        Initializes the StutterAnalyzer with configurable thresholds.
        
        Args:
            sample_rate (int): Audio sample rate (default: 16000 Hz for GCP STT)
            prolongation_threshold_sec (float): Duration threshold for prolongations (seconds)
            block_threshold_sec (float): Duration threshold for blocks/pauses (seconds)
            sentence_end_threshold_sec (float): More relaxed threshold for sentence-ending pauses (seconds)
            silence_top_db (int): dB threshold for silence detection (lower = more sensitive)
        """
        self.sr = sample_rate
        self.prolongation_threshold = prolongation_threshold_sec
        self.block_threshold = block_threshold_sec
        self.sentence_end_threshold = sentence_end_threshold_sec  # 문장 끝에는 더 관대하게
        self.silence_top_db = silence_top_db
        
        # Common Korean filler words that are often prolonged
        self.filler_words = {'음', '어', '그', '이', '저', '아'}
        
        # Sentence-ending punctuation marks
        self.sentence_endings = {'.', '!', '?', '。', '！', '？'}
    
    def _detect_repetitions(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes text segments for word repetitions.
        
        Args:
            transcript: Diarized transcript with text and timestamps
            
        Returns:
            List of detected repetitions with context
        """
        repetitions = []
        
        # Multiple patterns to catch different types of repetitions
        patterns = [
            # Pattern 1: Consecutive identical words with space: "나는 나는"
            re.compile(r'\b(\w+)\s+\1\b', re.UNICODE),
            # Pattern 2: Partial repetitions: "이 이제", "그 그니까" (1-2 characters repeated)
            re.compile(r'\b(\w{1,2})\s+\1\w+', re.UNICODE),
            # Pattern 3: Sound repetitions without space: "이-이제", "그-그니까"
            re.compile(r'\b(\w{1,3})-\1', re.UNICODE),
            # Pattern 4: Multiple consecutive same words: "음 음 음"
            re.compile(r'\b(\w+)(\s+\1){2,}', re.UNICODE),
        ]
        
        for segment in transcript:
            text = segment.get("text", "")
            speaker = segment.get("speaker", "UNKNOWN")
            start_time = segment.get("start", 0)
            
            if not text:
                continue
            
            # Track found positions to avoid duplicates
            found_positions = set()
            
            # Try each pattern
            for pattern_idx, pattern in enumerate(patterns):
                matches = list(pattern.finditer(text))
                
                for match in matches:
                    # Skip if we already found something at this position
                    match_pos = match.start()
                    if match_pos in found_positions:
                        continue
                    
                    found_positions.add(match_pos)
                    repeated_word = match.group(1)
                    
                    # Determine repetition type
                    if pattern_idx == 1:
                        rep_type = "partial_repetition"
                    elif pattern_idx == 2:
                        rep_type = "sound_repetition"
                    elif pattern_idx == 3:
                        rep_type = "multiple_repetition"
                    else:
                        rep_type = "repetition"
                    
                    repetitions.append({
                        "type": rep_type,
                        "word": repeated_word,
                        "full_match": match.group(0),
                        "speaker": speaker,
                        "timestamp": start_time,
                        "context": text,
                        "severity": "moderate" if pattern_idx in [1, 2] else "mild"
                    })
            
            # Additional check: look for same consecutive words across the entire text
            # Split by common separators and check for adjacent duplicates
            words = re.split(r'[\s,.\?!]+', text.lower())
            for i in range(len(words) - 1):
                if words[i] and words[i] == words[i + 1] and len(words[i]) > 1:
                    # Check if we haven't already caught this
                    word_pattern = re.compile(rf'\b{re.escape(words[i])}\b.*?\b{re.escape(words[i])}\b', re.IGNORECASE)
                    match = word_pattern.search(text)
                    if match and match.start() not in found_positions:
                        found_positions.add(match.start())
                        repetitions.append({
                            "type": "word_repetition",
                            "word": words[i],
                            "full_match": match.group(0),
                            "speaker": speaker,
                            "timestamp": start_time,
                            "context": text,
                            "severity": "mild"
                        })
        
        return repetitions
    
    def _detect_prolongations(self, audio_path: str, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes audio for prolonged sounds using word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            transcript: Diarized transcript with word-level timestamps
            
        Returns:
            List of detected prolongations
        """
        prolongations = []
        
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            for segment in transcript:
                words = segment.get("words", [])
                speaker = segment.get("speaker", "UNKNOWN")
                
                if not words:
                    continue
                
                for i, word_info in enumerate(words):
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0)
                    end = word_info.get("end", 0)
                    duration = end - start
                    
                    # Check if this word is unusually long
                    # Target: filler words or first word in segment (common stutter location)
                    is_filler = any(filler in word for filler in self.filler_words)
                    is_first_word = i == 0
                    
                    if (is_filler or is_first_word) and duration > self.prolongation_threshold:
                        # Extract audio segment for this word
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        word_audio = audio[start_sample:end_sample]
                        
                        # Calculate energy to confirm it's not silence
                        if len(word_audio) > 0:
                            rms_energy = np.sqrt(np.mean(word_audio**2))
                            
                            # Only flag if there's actual audio content
                            if rms_energy > 0.01:  # Threshold to ignore near-silence
                                prolongations.append({
                                    "type": "prolongation",
                                    "word": word,
                                    "duration": round(duration, 2),
                                    "speaker": speaker,
                                    "timestamp": start,
                                    "severity": "severe" if duration > self.prolongation_threshold * 1.5 else "moderate",
                                    "energy": round(float(rms_energy), 4)
                                })
        
        except Exception as e:
            print(f"⚠️ Warning: Could not analyze prolongations: {e}")
        
        return prolongations
    
    def _is_near_sentence_end(self, text: str, position_ratio: float) -> bool:
        """
        Check if the position is near a sentence ending.
        
        Args:
            text: The text segment
            position_ratio: Position in the segment (0.0 to 1.0)
            
        Returns:
            True if near sentence ending, False otherwise
        """
        if not text:
            return False
        
        # Check if text ends with sentence-ending punctuation
        text_stripped = text.strip()
        if text_stripped and text_stripped[-1] in self.sentence_endings:
            # If the pause is in the latter half of the segment, it's likely after sentence end
            return position_ratio > 0.5
        
        # Check for sentence-ending punctuation within the text
        for i, char in enumerate(text):
            if char in self.sentence_endings:
                char_position_ratio = i / len(text)
                # If the pause position is close to a sentence ending (within 20%)
                if abs(char_position_ratio - position_ratio) < 0.2:
                    return True
        
        return False
    
    def _detect_blocks(self, audio_path: str, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes audio for blocks (unnatural silences within speech segments).
        Uses context-aware thresholds:
        - Within sentence: 1.0s threshold (stricter)
        - Near sentence end: 1.5s threshold (more relaxed)
        
        Args:
            audio_path: Path to the audio file
            transcript: Diarized transcript with timestamps
            
        Returns:
            List of detected blocks
        """
        blocks = []
        
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            for segment in transcript:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment.get("text", "")
                
                # Extract segment audio
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) == 0:
                    continue
                
                # Use librosa to detect silent intervals within the segment
                # A block is an unnatural pause WITHIN a continuous speech segment
                non_silent_intervals = librosa.effects.split(
                    segment_audio, 
                    top_db=self.silence_top_db,
                    frame_length=2048,
                    hop_length=512
                )
                
                # If there are multiple non-silent intervals, there are silences between them
                if len(non_silent_intervals) > 1:
                    for i in range(len(non_silent_intervals) - 1):
                        # Calculate silence duration between intervals
                        silence_start_sample = non_silent_intervals[i][1]
                        silence_end_sample = non_silent_intervals[i + 1][0]
                        silence_duration = (silence_end_sample - silence_start_sample) / sr
                        
                        # Calculate position ratio in the segment (0.0 to 1.0)
                        position_in_segment = silence_start_sample / len(segment_audio) if len(segment_audio) > 0 else 0.5
                        
                        # Context-aware threshold selection
                        is_near_end = self._is_near_sentence_end(text, position_in_segment)
                        threshold = self.sentence_end_threshold if is_near_end else self.block_threshold
                        
                        # If silence is longer than threshold, it's a block
                        if silence_duration > threshold:
                            # Calculate timestamp relative to full audio
                            block_timestamp = start + (silence_start_sample / sr)
                            
                            blocks.append({
                                "type": "block",
                                "duration": round(silence_duration, 2),
                                "speaker": speaker,
                                "timestamp": round(block_timestamp, 2),
                                "segment_text": text,
                                "context": "sentence_end" if is_near_end else "mid_sentence",
                                "threshold_used": round(threshold, 2),
                                "severity": "severe" if silence_duration > threshold * 1.5 else "moderate"
                            })
        
        except Exception as e:
            print(f"⚠️ Warning: Could not analyze blocks: {e}")
        
        return blocks
    
    def analyze(self, audio_path: str, diarized_transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs the full stuttering analysis pipeline.
        
        Args:
            audio_path (str): The path to the full audio file
            diarized_transcript (List[Dict[str, Any]]): The diarized transcript with timestamps
            
        Returns:
            Dict containing lists of detected stuttering events and statistics
        """
        # 1. Detect repetitions from text (fast)
        repetitions = self._detect_repetitions(diarized_transcript)
        
        # 2. Detect prolongations from audio (slower)
        prolongations = self._detect_prolongations(audio_path, diarized_transcript)
        
        # 3. Detect blocks from audio (slower)
        blocks = self._detect_blocks(audio_path, diarized_transcript)
        
        # Calculate statistics
        total_events = len(repetitions) + len(prolongations) + len(blocks)
        
        # Calculate overall fluency percentage
        # This is a simple heuristic: fewer stutter events = higher fluency
        total_segments = len(diarized_transcript)
        affected_segments = len(set(
            [r.get("timestamp") for r in repetitions] +
            [p.get("timestamp") for p in prolongations] +
            [b.get("timestamp") for b in blocks]
        ))
        
        fluency_percentage = 0
        if total_segments > 0:
            fluency_percentage = ((total_segments - min(affected_segments, total_segments)) / total_segments) * 100
        
        return {
            "repetitions": repetitions,
            "prolongations": prolongations,
            "blocks": blocks,
            "statistics": {
                "total_events": total_events,
                "repetition_count": len(repetitions),
                "prolongation_count": len(prolongations),
                "block_count": len(blocks),
                "fluency_percentage": round(fluency_percentage, 1),
                "affected_segments": affected_segments,
                "total_segments": total_segments
            }
        }
    
    def format_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Formats analysis results into a human-readable summary.
        
        Args:
            analysis_results: Results from analyze() method
            
        Returns:
            Formatted string summary
        """
        stats = analysis_results.get("statistics", {})
        repetitions = analysis_results.get("repetitions", [])
        prolongations = analysis_results.get("prolongations", [])
        blocks = analysis_results.get("blocks", [])
        
        summary = []
        summary.append("=" * 60)
        summary.append("말더듬 분석 결과")
        summary.append("=" * 60)
        summary.append(f"\n유창성 점수: {stats.get('fluency_percentage', 0):.1f}%")
        summary.append(f"총 {stats.get('total_events', 0)}개의 말더듬 이벤트 검출\n")
        
        # Repetitions
        if repetitions:
            summary.append(f"🔁 반복 (Repetitions): {len(repetitions)}회")
            
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
                summary.append(f"   • {type_name}: {count}회")
            
            # Show examples
            summary.append("\n   예시:")
            for rep in repetitions[:5]:  # Show first 5
                summary.append(f"   - [{rep.get('timestamp', 0):.1f}s] {rep.get('speaker')}: '{rep.get('full_match')}'")
            if len(repetitions) > 5:
                summary.append(f"   ... 그 외 {len(repetitions) - 5}회 더")
        else:
            summary.append("🔁 반복 (Repetitions): 없음")
        
        # Prolongations
        summary.append("")
        if prolongations:
            summary.append(f"⏱️ 연장 (Prolongations): {len(prolongations)}회")
            for prol in prolongations[:3]:
                summary.append(f"   - [{prol.get('timestamp', 0):.1f}s] {prol.get('speaker')}: '{prol.get('word')}' ({prol.get('duration')}초)")
            if len(prolongations) > 3:
                summary.append(f"   ... 그 외 {len(prolongations) - 3}회 더")
        else:
            summary.append("⏱️ 연장 (Prolongations): 없음")
        
        # Blocks
        summary.append("")
        if blocks:
            summary.append(f"🚫 막힘 (Blocks): {len(blocks)}회")
            for block in blocks[:3]:
                summary.append(f"   - [{block.get('timestamp', 0):.1f}s] {block.get('speaker')}: {block.get('duration')}초 침묵")
            if len(blocks) > 3:
                summary.append(f"   ... 그 외 {len(blocks) - 3}회 더")
        else:
            summary.append("🚫 막힘 (Blocks): 없음")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)

