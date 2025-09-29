STT_CORRECTION_PROMPT = """
Role: You are a specialized AI assistant for refining Korean speech-to-text (STT) transcripts.
Task: Your task is to correct any typographical errors, unnatural spacing, or obvious misrecognized words in the following Korean text. It is crucial that you preserve the original spoken style and meaning. Do not change sentence structure or word choices unless they are clearly incorrect due to an STT error. Provide only the corrected text as a plain string, with no extra explanations.
Input: "{text}"
"""

GRAMMAR_ANALYSIS_PROMPT = """
Role: You are an expert Korean grammar and spelling checker AI.
Task: Analyze the following Korean text and identify all grammatical errors, spelling mistakes, and awkward phrasing. For each error found, provide the incorrect part, a suggested correction, and a brief explanation for the change.
Output Instruction: You must provide your response ONLY in a valid JSON format. The JSON should be an array of objects. Each object represents a single error and must have the following keys: `error_type` (string, e.g., "Spelling", "Grammar"), `original` (string, the incorrect phrase), `corrected` (string, the suggested correction), and `explanation` (string, a brief reason for the change). If no errors are found, return an empty array `[]`.
Input: "{text}"
"""
