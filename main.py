import os
import logging
from dotenv import load_dotenv

from src.session import HabitLinkSession

# Suppress library warnings
logging.getLogger('pyannote').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
os.environ["PYTORCH_SUPPRESS_DEPRECATION_WARNINGS"] = "1"
os.environ["PL_SUPPRESS_FORK"] = "1"


def check_environment() -> bool:
    """
    Verify that all required environment variables and credentials are set.
    
    Returns:
        bool: True if environment is properly configured, False otherwise.
    """
    # Check for GROQ API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        print("Please set your GROQ_API_KEY to use LLM-based analysis features.")
        return False
    
    # Check Google Cloud credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.path.exists("gcp_credentials.json"):
        print("❌ Error: Google Cloud credentials not found.")
        print("Please either:")
        print("  1. Set GOOGLE_APPLICATION_CREDENTIALS in your .env file, or")
        print("  2. Run 'gcloud auth application-default login'")
        return False
    
    return True


def print_welcome():
    """Print welcome banner."""
    print("="*60)
    print("🎯 HabitLink: AI-Powered Korean Speech Habit Correction System")
    print("="*60)
    print("\n실시간 음성 분석 세션을 시작합니다.")


def main():
    """Main entry point for the HabitLink application."""
    # Load environment variables
    load_dotenv()
    
    # Check environment
    if not check_environment():
        return
    
    # Print welcome message
    print_welcome()
    
    # Ask about UI visualization
    print("\n실시간 음성 시각화 UI를 활성화하시겠습니까?")
    ui_choice = input("UI 활성화 (Y/n): ").strip().lower()
    enable_ui = ui_choice != 'n'
    
    if enable_ui:
        print("✅ UI 시각화가 활성화됩니다.")
    else:
        print("ℹ️  콘솔 모드로 실행됩니다.")
    
    # Create and run session
    session = HabitLinkSession()
    session.run(enable_ui=enable_ui)
    
    # Goodbye message
    print("\n\n감사합니다. HabitLink를 사용해 주셔서 감사합니다! 👋")


if __name__ == "__main__":
    main()
