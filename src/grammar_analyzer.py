import os
import json
from groq import Groq
from typing import List, Dict, Any
from .llm_prompts import STT_CORRECTION_PROMPT, GRAMMAR_ANALYSIS_PROMPT

class GrammarAnalyzer:
    """
    Analyzes text for grammatical errors using the Groq LLM API.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        """
        Initializes the GrammarAnalyzer with a Groq client.
        Requires the GROQ_API_KEY environment variable to be set.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        """A helper method to call the LLM API."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return ""

    def _get_structured_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        Performs the two-step LLM process to get structured grammatical errors.
        """
        # Step 1: Correct raw STT output
        correction_prompt = STT_CORRECTION_PROMPT.format(text=text)
        corrected_text = self._call_llm(correction_prompt)

        if not corrected_text:
            return []

        # Step 2: Find grammatical errors in the corrected text
        grammar_prompt = GRAMMAR_ANALYSIS_PROMPT.format(text=corrected_text)
        response_str = self._call_llm(grammar_prompt)
        
        try:
            # Clean the response string in case the LLM includes markdown formatting
            clean_response = response_str.strip().replace('```json', '').replace('```', '')
            errors = json.loads(clean_response)
            return errors if isinstance(errors, list) else []
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Failed to parse LLM response into JSON. Response: {response_str}")
            return []

    def analyze(self, diarized_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes each segment of a diarized transcript for grammatical errors.
        """
        analysis_results = []
        for segment in diarized_transcript:
            text = segment.get("text", "")
            if not text:
                continue
            
            errors = self._get_structured_errors(text)
            if errors:
                # Associate each found error with its original context
                for error in errors:
                    analysis_results.append({
                        "speaker": segment.get("speaker"),
                        "timestamp": segment.get("start"),
                        "context": text,
                        "error_details": error
                    })
        return analysis_results
