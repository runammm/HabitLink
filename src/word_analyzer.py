import re
from typing import List, Dict, Any

class WordAnalyzer:
    """
    Analyzes transcribed text to detect user-defined keywords at the word level.
    """
    def __init__(self):
        """
        Initializes the WordAnalyzer.
        """
        pass

    def analyze(self, diarized_transcript: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Scans a diarized transcript for occurrences of specified keywords, using word-level timestamps.

        Args:
            diarized_transcript (List[Dict[str, Any]]): 
                The output from the SpeakerDiarizer, where each dict contains
                'text', 'speaker', and a 'words' list with word-level timestamps.
            keywords (List[str]): 
                A list of keywords to search for (case-insensitive).

        Returns:
            List[Dict[str, Any]]: 
                A list of dictionaries for each found keyword, including the keyword,
                speaker, and its precise timestamp.
        """
        if not keywords:
            return []

        found_keywords = []
        # Prepare a set of lowercased keywords for efficient, case-insensitive matching.
        search_keywords = {kw.strip().lower() for kw in keywords}

        for segment in diarized_transcript:
            speaker = segment.get("speaker")
            words = segment.get("words", [])

            if not words:
                continue
            
            for word_info in words:
                word = word_info.get("word", "").strip().lower()
                
                # Direct matching is now possible since we iterate word by word.
                if word in search_keywords:
                    found_keywords.append({
                        "keyword": word_info.get("word"), # Store the original casing
                        "speaker": speaker,
                        "timestamp": word_info.get("start") # Use the word's precise start time
                    })

        return found_keywords
