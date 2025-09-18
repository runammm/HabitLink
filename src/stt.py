from abc import ABC
import whisper
from groq import Groq
from dotenv import load_dotenv
import os

class STT(ABC):
    def __init__(self, model_name: str):
        pass

    def transcribe(self, input_path: str, language: str = "ko") -> str:
        pass


class STT_Whisper(STT):
    def __init__(self, model_name: str = "base"):
        """
        OpenAI Whisper를 사용한 음성 인식 클래스
        
        Args:
            model_name (str): Whisper 모델 이름 ("tiny", "base", "small", "medium", "large")
        """
        super().__init__(model_name)
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, input_path: str, language: str = "ko") -> str:
        """
        음성 파일을 텍스트로 변환
        
        Args:
            input_path (str): 음성 파일 경로
            language (str): 인식할 언어 코드 (기본값: "ko" - 한국어)
            
        Returns:
            str: 변환된 텍스트
        """
        try:
            result = self.model.transcribe(input_path, language=language)
            return result["text"].strip()
        except Exception as e:
            print(f"음성 인식 중 오류 발생: {e}")
            return ""
    
    class STT_Groq(STT):
        def __init__(self, model_name: str = "whisper-large-v3"):
            super().__init__(model_name)
            self.model = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
        def transcribe(self, input_path: str, language: str = "ko") -> str:
            return self.model.transcribe(input_path, language=language)


if __name__ == "__main__":
    stt = STT_Whisper()
    print(stt.transcribe("temp.wav", language="ko"))
