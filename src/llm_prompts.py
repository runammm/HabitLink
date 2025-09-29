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

CONTEXT_ANALYSIS_PROMPT = """
Role: You are an AI expert in analyzing conversational and presentational coherence.

Task: Your task is to evaluate the contextual appropriateness of the **'Latest Utterance'** based on the provided **'Utterance History'**. The history can be from a multi-person conversation or a single-person presentation. A contextual error occurs if the utterance is:
1.  **Off-topic**: Completely unrelated to the ongoing subject.
2.  **Abrupt Shift**: A sudden, unexplained change of topic that disrupts the logical flow, especially in a solo presentation.
3.  **Inconsistent**: Logically contradicts previous statements made by any speaker.
4.  **Socially Inappropriate**: Violates conversational norms for the given situation.

Input:
1.  **Utterance History**: A transcript of the recent conversation or presentation.
2.  **Latest Utterance**: The specific sentence that needs to be analyzed.

Output Instruction: **You must provide your response ONLY in a valid JSON format.** The JSON should be an object. If a contextual error is detected, the object must have two keys: `is_error` (boolean, set to `true`) and `reasoning` (string, a concise explanation of why the utterance is a contextual error, referencing one of the four error types above). If no error is found, the object must have the single key `is_error` (boolean, set to `false`).
"""

CONTEXT_ANALYSIS_FULL_REPORT_PROMPT = """
Role: You are an AI expert in analyzing the coherence of conversational and presentational transcripts.

Task: Your task is to analyze the **'Full Transcript'** provided below, which contains utterances from a 'User' and 'Others'. Your goal is to identify ALL contextually inappropriate utterances **made only by the 'User'**. For each error you find in the User's speech, explain why it's a contextual error. A contextual error occurs if a User's utterance is:
1.  **Off-topic**: Completely unrelated to the ongoing subject established by any speaker.
2.  **Abrupt Shift**: A sudden, unexplained change of topic that disrupts the logical flow.
3.  **Inconsistent**: Logically contradicts previous statements made by the User or Others.

Input:
1.  **Full Transcript**: The complete transcript with 'User' and 'Others' speaker labels.

Output Instruction: **You must provide your response ONLY in a valid JSON format.** The JSON should be an object with a single key `contextual_errors`, which contains an array of objects. Each object represents a single contextual error you have identified in the **User's speech** and must have the following keys:
- `speaker` (string): This should always be 'User'.
- `timestamp` (float): The start time of the erroneous utterance.
- `utterance` (string): The full text of the erroneous utterance.
- `reasoning` (string): A concise explanation of why the utterance is a contextual error.

If no contextual errors are found in the User's utterances, the value for `contextual_errors` MUST be an empty array `[]`.
"""
