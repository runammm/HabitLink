from groq import Groq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv('.env', override=True)

# Groq 클라이언트 초기화
# .env 파일에 GROQ_API_KEY가 설정되어 있어야 합니다.
try:
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
except KeyError:
    print("="*50)
    print("⚠️ GROQ_API_KEY가 설정되지 않았습니다.")
    print(".env 파일에 'GROQ_API_KEY=YOUR_API_KEY' 형식으로 키를 추가해주세요.")
    print("="*50)
    client = None

async def stt(file_path: str) -> str:
    """
    Groq API를 사용하여 오디오 파일을 텍스트로 변환합니다.

    Args:
        file_path (str): 변환할 오디오 파일의 경로.

    Returns:
        str: 변환된 텍스트. API 호출에 실패하면 빈 문자열을 반환합니다.
    """
    if not client:
        print("❌ Groq 클라이언트가 초기화되지 않아 STT를 진행할 수 없습니다.")
        return ""

    def _blocking_stt_call() -> str:
        """Groq API에 대한 동기적인 호출을 처리하는 내부 함수."""
        try:
            print(f"Groq STT 시작: {os.path.basename(file_path)}")
            with open(file_path, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), audio_file.read()),
                    model="whisper-large-v3",
                    language="ko",
                )
            print(f"Groq STT 완료: {transcription.text}")
            return transcription.text
        except Exception as e:
            print(f"❌ Groq STT API 오류: {e}")
            return ""
        finally:
            # 작업 완료 후 임시 오디오 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)

    # 동기 함수를 별도의 스레드에서 실행하여 비동기 이벤트 루프를 막지 않도록 함
    return await asyncio.to_thread(_blocking_stt_call)