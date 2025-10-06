import os
import json
import asyncio
from groq import AsyncGroq
from typing import List, Dict, Any
from .llm_prompts import COMPREHENSIVE_TEXT_ANALYSIS_PROMPT

class TextAnalyzer:
    """
    Performs comprehensive, asynchronous text analysis for grammar and context using a single LLM call.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        """
        Initializes the TextAnalyzer with an async Groq client.
        Requires the GROQ_API_KEY environment variable to be set.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    def _format_transcript_for_prompt(self, transcript: List[Dict[str, Any]]) -> str:
        """Formats the transcript list into an indexed, readable string for the LLM prompt."""
        return "\n".join(
            [f"[{i}] {segment.get('speaker', 'UNKNOWN')}: {segment.get('text', '')}" for i, segment in enumerate(transcript)]
        )

    async def analyze(self, diarized_transcript: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyzes the entire diarized transcript for grammatical and contextual errors in a single async call.
        """
        if not diarized_transcript:
            return {"grammar_errors": [], "context_errors": []}

        full_transcript_text = self._format_transcript_for_prompt(diarized_transcript)
        prompt = COMPREHENSIVE_TEXT_ANALYSIS_PROMPT.format(transcript=full_transcript_text)
        
        default_return = {"grammar_errors": [], "context_errors": []}
        response_content = ""

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.choices[0].message.content
            analysis_results = json.loads(response_content)

            final_results = {"grammar_errors": [], "context_errors": []}

            for error in analysis_results.get("grammar_errors", []):
                idx = error.get("index")
                if idx is not None and 0 <= idx < len(diarized_transcript):
                    segment = diarized_transcript[idx]
                    final_results["grammar_errors"].append({
                        "speaker": segment.get("speaker", "UNKNOWN"),
                        "timestamp": segment.get("start", 0),
                        "error_details": error
                    })

            for error in analysis_results.get("context_errors", []):
                idx = error.get("index")
                if idx is not None and 0 <= idx < len(diarized_transcript):
                    segment = diarized_transcript[idx]
                    final_results["context_errors"].append({
                        "speaker": segment.get("speaker", "UNKNOWN"),
                        "timestamp": segment.get("start", 0),
                        "utterance": error.get("utterance"),
                        "reasoning": error.get("reasoning")
                    })
            
            return final_results

        except json.JSONDecodeError:
            print(f"Warning: Failed to parse LLM response into JSON. Response: {response_content}")
            return default_return
        except Exception as e:
            print(f"Error during comprehensive text analysis: {e}")
            return default_return
