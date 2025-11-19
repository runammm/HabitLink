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
                'text', 'speaker', and optionally a 'words' list with word-level timestamps.
            keywords (List[str]): 
                A list of keywords to search for (case-insensitive).
                For Korean profanity, also performs substring matching.

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
        
        # Track found positions to avoid duplicates
        found_positions = set()

        for segment in diarized_transcript:
            speaker = segment.get("speaker")
            words = segment.get("words", [])

            # Get text for substring matching (important for Korean profanity)
            text = segment.get("text", "")
            text_lower = text.lower()
            timestamp = segment.get("start", 0)
            
            # If word-level timestamps are available, use them
            if words:
                for word_info in words:
                    word = word_info.get("word", "").strip().lower()
                    word_timestamp = word_info.get("start", timestamp)
                    position_key = f"{word_timestamp}_{word}"
                    
                    # Direct word matching
                    if word in search_keywords and position_key not in found_positions:
                        found_positions.add(position_key)
                        found_keywords.append({
                            "keyword": word_info.get("word"),  # Store the original casing
                            "speaker": speaker,
                            "timestamp": word_timestamp  # Use the word's precise start time
                        })
                    
                    # Substring matching for Korean (e.g., "시발" in "시발새끼", "개" and "좆같은" in "좆같은게")
                    # Note: We must check ALL keywords, not just the first match (no break)
                    # A single word can contain multiple profanities (e.g., "좆같은게" contains both "좆같은" and "개")
                    else:
                        for kw in search_keywords:
                            # Create unique position key for each keyword in this word
                            position_key_for_kw = f"{word_timestamp}_{word}_{kw}"
                            if kw in word and position_key_for_kw not in found_positions:
                                found_positions.add(position_key_for_kw)
                                found_keywords.append({
                                    "keyword": kw,
                                    "speaker": speaker,
                                    "timestamp": word_timestamp
                                })
                                # NO break - continue checking other keywords in the same word
            else:
                # Fallback: Parse the text directly if word-level timestamps are not available
                # Split text by whitespace and punctuation to extract words
                text_words = re.split(r'[\s,.\?!;:]+', text)
                
                for word in text_words:
                    word_lower = word.strip().lower()
                    position_key = f"{timestamp}_{word_lower}"
                    
                    # Direct word matching
                    if word_lower and word_lower in search_keywords and position_key not in found_positions:
                        found_positions.add(position_key)
                        # Find the original casing in the text
                        original_match = None
                        for kw in keywords:
                            if kw.strip().lower() == word_lower:
                                original_match = kw.strip()
                                break
                        
                        found_keywords.append({
                            "keyword": original_match or word.strip(),
                            "speaker": speaker,
                            "timestamp": timestamp
                        })
                    
                    # Substring matching for Korean profanity (e.g., "시발" in "시발새끼", "개" and "좆같은" in "좆같은게")
                    # Note: We must check ALL keywords, not just the first match (no break)
                    # A single word can contain multiple profanities (e.g., "좆같은게" contains both "좆같은" and "개")
                    elif word_lower:
                        for kw in search_keywords:
                            # Create unique position key for each keyword in this word
                            position_key_for_kw = f"{timestamp}_{word_lower}_{kw}"
                            if kw in word_lower and position_key_for_kw not in found_positions:
                                found_positions.add(position_key_for_kw)
                                found_keywords.append({
                                    "keyword": kw,
                                    "speaker": speaker,
                                    "timestamp": timestamp
                                })
                                # NO break - continue checking other keywords in the same word

        return found_keywords
