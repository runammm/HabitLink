from typing import List

def load_profanity_list(path: str = ".data/profanity_list_ko.txt") -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, and filter out empty lines
            profanities = [line.strip() for line in f if line.strip()]
        return profanities
    except FileNotFoundError:
        print(f"Warning: Profanity file not found at {path}. Profanity detection will be skipped.")
        return []
