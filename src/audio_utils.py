import numpy as np
import librosa


def detect_speech_duration(audio_path: str, sample_rate: int = 16000, 
                          top_db: int = 30, frame_length: int = 2048, 
                          hop_length: int = 512) -> float:
    """
    Detect actual speech duration in an audio file using librosa's VAD.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Sample rate to use for loading audio
        top_db: Threshold in decibels below reference to consider as silence
        frame_length: Frame length for non-silent interval detection
        hop_length: Hop length for non-silent interval detection
    
    Returns:
        float: Total duration of speech in seconds (excludes silence)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Detect non-silent intervals
        # librosa.effects.split returns intervals where audio is above threshold
        intervals = librosa.effects.split(y, top_db=top_db, 
                                         frame_length=frame_length, 
                                         hop_length=hop_length)
        
        # Calculate total speech duration
        total_speech_samples = 0
        for start_sample, end_sample in intervals:
            total_speech_samples += (end_sample - start_sample)
        
        # Convert samples to seconds
        speech_duration = total_speech_samples / sr
        
        return max(0.1, speech_duration)  # Ensure at least 0.1 seconds
        
    except Exception as e:
        print(f"⚠️ Error detecting speech duration: {e}")
        # Return a default value if detection fails
        return 1.0


def detect_speech_segments(audio_data: np.ndarray, sample_rate: int = 16000,
                           top_db: int = 30, frame_length: int = 2048,
                           hop_length: int = 512) -> tuple:
    """
    Detect speech segments from audio data (numpy array).
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        top_db: Threshold in decibels below reference to consider as silence
        frame_length: Frame length for non-silent interval detection
        hop_length: Hop length for non-silent interval detection
    
    Returns:
        tuple: (total_speech_duration, speech_intervals)
            - total_speech_duration: Total duration of speech in seconds
            - speech_intervals: List of (start_time, end_time) tuples in seconds
    """
    try:
        # Detect non-silent intervals
        intervals = librosa.effects.split(audio_data, top_db=top_db,
                                         frame_length=frame_length,
                                         hop_length=hop_length)
        
        # Convert sample indices to time
        speech_intervals = []
        total_speech_samples = 0
        
        for start_sample, end_sample in intervals:
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            speech_intervals.append((start_time, end_time))
            total_speech_samples += (end_sample - start_sample)
        
        total_speech_duration = total_speech_samples / sample_rate
        
        return max(0.1, total_speech_duration), speech_intervals
        
    except Exception as e:
        print(f"⚠️ Error detecting speech segments: {e}")
        return 1.0, []


def calculate_speech_rate(text: str, speech_duration: float) -> tuple:
    """
    Calculate speech rate (WPM) from text and speech duration.
    
    Args:
        text: Transcribed text
        speech_duration: Actual speech duration in seconds (excluding silence)
    
    Returns:
        tuple: (word_count, wpm)
            - word_count: Number of words in text
            - wpm: Words per minute
    """
    word_count = len(text.split())
    
    if speech_duration <= 0:
        return word_count, 0.0
    
    wpm = (word_count / speech_duration) * 60.0
    
    return word_count, round(wpm, 2)

