from typing import List, Dict, Any, Optional

class SpeechRateAnalyzer:
    """
    Analyzes diarized transcript segments to calculate speech rate metrics (WPS and WPM).
    """
    def __init__(self, tokenizer_method: str = 'simple', target_wpm: Optional[float] = None):
        """
        Initializes the SpeechRateAnalyzer.

        Args:
            tokenizer_method (str): The method for counting words. 'simple' for whitespace splitting.
                                    (Future versions could support 'mecab', 'okt', etc.)
            target_wpm (Optional[float]): The user's target speaking rate in WPM. 
                                          If set, analysis will include comparisons to this target.
        """
        self.tokenizer_method = tokenizer_method
        self.target_wpm = target_wpm

    def _count_words(self, text: str) -> int:
        """
        Counts the number of words in a text string based on the chosen method.
        """
        if self.tokenizer_method == 'simple':
            return len(text.split())
        # Add other tokenization methods here in the future
        else:
            # Default to simple splitting if method is unknown
            return len(text.split())

    def set_target_wpm(self, target_wpm: float):
        """
        Sets the target WPM for speech rate comparison.
        
        Args:
            target_wpm (float): The user's desired speaking rate in words per minute.
        """
        self.target_wpm = target_wpm

    def analyze(self, diarized_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates WPS and WPM for each segment in a diarized transcript.

        Args:
            diarized_transcript (List[Dict[str, Any]]): 
                The output from the SpeakerDiarizer.

        Returns:
            List[Dict[str, Any]]: 
                A list of dictionaries, each containing speech rate analysis for a segment.
                Example: [{'speaker': 'SPEAKER_00', 'start': 10.5, 'end': 15.0, 'duration': 4.5, 
                           'word_count': 15, 'wps': 3.33, 'wpm': 200.0, 'comparison': 'too_fast'}]
        """
        rate_analysis_results = []

        for segment in diarized_transcript:
            start_time = segment.get("start")
            end_time = segment.get("end")
            text = segment.get("text", "")

            if start_time is None or end_time is None or not text:
                continue

            duration = end_time - start_time
            if duration <= 0:
                continue
            
            word_count = self._count_words(text)
            wps = word_count / duration
            wpm = wps * 60

            result = {
                "speaker": segment.get("speaker"),
                "start": start_time,
                "end": end_time,
                "duration": round(duration, 2),
                "word_count": word_count,
                "wps": round(wps, 2),
                "wpm": round(wpm, 2)
            }

            # Add comparison if target_wpm is set
            if self.target_wpm is not None:
                # Allow 20% tolerance
                tolerance = self.target_wpm * 0.2
                if wpm > self.target_wpm + tolerance:
                    result["comparison"] = "too_fast"
                elif wpm < self.target_wpm - tolerance:
                    result["comparison"] = "too_slow"
                else:
                    result["comparison"] = "good"

            rate_analysis_results.append(result)

        return rate_analysis_results
