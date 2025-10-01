import os
import json
from groq import Groq
from typing import List, Dict, Any
from .llm_prompts import CONTEXT_ANALYSIS_PROMPT

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

    def analyze(self, diarized_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes each user utterance in the context of the preceding conversation.
        NOTE: For file-based analysis, this simulates a real-time check for every user utterance.
        """
        analysis_results = []
        for i, segment in enumerate(diarized_transcript):
            # We only analyze the user's speech for contextual errors.
            if segment.get("speaker") != self.user_speaker_id:
                continue

            # Context requires at least one preceding utterance.
            if i == 0:
                continue

            conversation_history = diarized_transcript[:i]
            latest_utterance = segment

            prompt = CONTEXT_ANALYSIS_PROMPT.format(
                history=self._format_history_for_prompt(conversation_history),
                utterance=latest_utterance.get('text', '')
            )

            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.choices[0].message.content
                analysis = json.loads(response_content)

                if analysis.get("is_error"):
                    analysis_results.append({
                        "contextual_error": True,
                        "reasoning": analysis.get("reasoning"),
                        "utterance": latest_utterance.get('text', ''),
                        "timestamp": latest_utterance.get('start'),
                        "speaker": latest_utterance.get('speaker')
                    })
            except Exception as e:
                print(f"Error during context analysis for segment {i}: {e}")
                continue
                
        return analysis_results
