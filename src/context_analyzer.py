import os
import json
from groq import Groq
from typing import List, Dict, Any, Optional
from .llm_prompts import CONTEXT_ANALYSIS_PROMPT, CONTEXT_ANALYSIS_FULL_REPORT_PROMPT

class ContextAnalyzer:
    """
    Analyzes conversation history to detect contextual errors in a user's speech.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b", user_speaker_id: str = "User"):
        """
        Initializes the ContextAnalyzer.
        
        Args:
            model (str): The LLM model to use via Groq.
            user_speaker_id (str): The identifier for the primary user whose speech is being analyzed.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.user_speaker_id = user_speaker_id

    def _format_history_for_prompt(self, transcript: List[Dict[str, Any]]) -> str:
        """Formats the transcript list into a readable string for the LLM prompt."""
        return "\n".join([f"{segment.get('speaker', 'UNKNOWN')}: {segment.get('text', '')}" for segment in transcript])

    def analyze_full_transcript(self, diarized_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes the entire transcript for all contextual errors to generate a post-session report.
        """
        if not diarized_transcript or len(diarized_transcript) < 2:
            return []

        formatted_transcript = self._format_history_for_prompt(diarized_transcript)
        
        prompt_input = f"Full Transcript:\n{formatted_transcript}"
        full_prompt = f"{CONTEXT_ANALYSIS_FULL_REPORT_PROMPT}\n\n{prompt_input}"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=self.model,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.choices[0].message.content
            analysis = json.loads(response_content)
            
            return analysis.get("contextual_errors", []) if isinstance(analysis, dict) else []

        except Exception as e:
            print(f"Error during full context analysis: {e}")
            return []

    def analyze_latest_utterance(self, diarized_transcript: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyzes the latest user utterance in the context of the full conversation for real-time feedback.

        Args:
            diarized_transcript (List[Dict[str, Any]]): The full conversation transcript.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the analysis result if a user utterance is found,
                                      otherwise None. 
                                      Example: {'contextual_error': True, 'reasoning': '...', 'erroneous_sentence': '...'}
        """
        if not diarized_transcript or len(diarized_transcript) < 2:
            return None

        latest_utterance = diarized_transcript[-1]
        
        if latest_utterance.get("speaker") != self.user_speaker_id:
             return None

        conversation_history = diarized_transcript[:-1]
        
        prompt_input = f"""
        Utterance History:
        {self._format_history_for_prompt(conversation_history)}

        Latest Utterance:
        {latest_utterance.get('text', '')}
        """
        
        full_prompt = f"{CONTEXT_ANALYSIS_PROMPT}\n\n{prompt_input}"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=self.model,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.choices[0].message.content
            analysis = json.loads(response_content)

            if analysis.get("is_error"):
                return {
                    "contextual_error": True,
                    "reasoning": analysis.get("reasoning"),
                    "erroneous_sentence": latest_utterance.get('text', ''),
                    "timestamp": latest_utterance.get('start')
                }
            return {"contextual_error": False}
        except Exception as e:
            print(f"Error during context analysis: {e}")
            return None
