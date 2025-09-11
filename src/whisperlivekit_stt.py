import asyncio
import os
import subprocess
import tempfile
import signal
from typing import Optional
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
# 이 모듈이 임포트될 때 바로 실행됩니다.
load_dotenv()


class WhisperLiveSTT:
    """
    WhisperLiveKit 실시간 스트리밍 서버를 제어하는 클래스.
    """
    def __init__(self, model: str = "base", language: str = "ko", port: int = 8000, diarization: bool = False):
        self.model = model
        self.language = language
        self.port = port
        self.diarization = diarization
        self.server_process = None
        self.server_running = False
    
    async def start_server(self) -> bool:
        """
        WhisperLiveKit 서버를 백그라운드에서 시작합니다.
        """
        if self.server_running:
            print("WhisperLiveKit 서버가 이미 실행 중입니다.")
            return True
        
        try:
            cmd = [
                "whisperlivekit-server",
                "--model", self.model,
                "--lan", self.language,
                "--port", str(self.port),
                "--host", "localhost"
            ]
            
            if self.diarization:
                cmd.append("--diarization")
                print("화자 구분 기능이 활성화됩니다.")
            
            print(f"WhisperLiveKit 서버 시작 중... (모델: {self.model}, 언어: {self.language})")
            print(f"실행 명령어: {' '.join(cmd)}")
            
            self.server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # 서버가 시작될 때까지 최대 15초 대기
            for i in range(5):
                await asyncio.sleep(3)
                
                if self.server_process.returncode is not None:
                    stderr = await self.server_process.stderr.read()
                    print(f"❌ 서버 프로세스가 예기치 않게 종료되었습니다. 오류:\n{stderr.decode()}")
                    return False
                
                # 포트 활성화 확인
                try:
                    conn = asyncio.open_connection('localhost', self.port)
                    _, writer = await asyncio.wait_for(conn, timeout=1.0)
                    writer.close()
                    await writer.wait_closed()
                    self.server_running = True
                    print(f"✅ WhisperLiveKit 서버가 포트 {self.port}에서 정상 작동 중입니다.")
                    print(f"🌐 브라우저에서 http://localhost:{self.port} 에 접속하세요.")
                    return True
                except (ConnectionRefusedError, asyncio.TimeoutError):
                    print(f"⏳ 서버 준비 중... ({i+1}/5)")
                    continue
            
            print("❌ 서버 시작 시간 초과 (15초).")
            return False
                
        except FileNotFoundError:
            print("❌ 'whisperlivekit-server'를 찾을 수 없습니다. 설치를 확인하세요.")
            print("   pip install whisperlivekit")
            return False
        except Exception as e:
            print(f"❌ 서버 시작 중 오류 발생: {e}")
            return False
    
    async def stop_server(self):
        """
        WhisperLiveKit 서버를 종료합니다.
        """
        if self.server_process and self.server_running:
            print("WhisperLiveKit 서버를 종료합니다...")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                await asyncio.wait_for(self.server_process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
            
            self.server_running = False
            print("✅ 서버가 종료되었습니다.")


async def stt(file_path: str, enable_diarization: bool = False) -> str:
    """
    오디오 파일을 텍스트로 변환하는 간단한 함수. 화자 분리는 지원하지 않습니다.
    """
    if enable_diarization:
        print("⚠️ 'stt' 함수는 화자 분리를 지원하지 않습니다. 'stt_with_speakers'를 사용하세요.")

    output_txt_path = ""
    try:
        # whisper CLI가 생성할 .txt 파일 경로를 미리 계산
        file_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_txt_path = os.path.join(file_dir, f"{base_name}.txt")

        cmd = [
            "whisper", file_path, 
            "--model", "base", 
            "--language", "ko", 
            "--output_format", "txt",
            "--output_dir", file_dir  # 출력 디렉토리를 명시적으로 지정
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            if os.path.exists(output_txt_path):
                with open(output_txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                print(f"❌ Whisper 변환 후 출력 파일({output_txt_path})을 찾을 수 없습니다.")
                return ""
        else:
            print(f"Whisper 변환 실패: {stderr.decode()}")
            return ""
    except Exception as e:
        print(f"음성 변환 중 오류: {e}")
        return ""
    finally:
        # 원본 오디오 파일과 생성된 txt 파일 모두 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        if output_txt_path and os.path.exists(output_txt_path):
            os.remove(output_txt_path)


async def stt_with_speakers(file_path: str) -> dict:
    """
    Whisper CLI의 --diarize 옵션을 사용하여 화자 분리가 포함된 음성-텍스트 변환을 수행합니다.
    이 기능을 사용하려면 HUGGING_FACE_HUB_TOKEN 환경 변수가 설정되어 있어야 합니다.
    """
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
    
    # Hugging Face 토큰 확인
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("="*50)
        print("⚠️ HUGGING_FACE_HUB_TOKEN 환경 변수가 설정되지 않았습니다.")
        print("화자 분리 기능을 사용하려면 Hugging Face 인증 토큰이 필요합니다.")
        print("https://huggingface.co/settings/tokens 에서 토큰을 발급받은 후,")
        print("`.env` 파일에 HUGGING_FACE_HUB_TOKEN=your_token 형식으로 추가해주세요.")
        print("="*50)
        return {}

    try:
        cmd = [
            "whisper", file_path,
            "--model", "base",
            "--language", "ko",
            "--diarize",
            "--output_format", "txt",
            "--output_dir", output_dir
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode()
            # 토큰 관련 오류 메시지인지 확인
            if "Authentication Error" in error_message or "Gated model" in error_message:
                print("❌ Hugging Face 인증 오류가 발생했습니다. 토큰이 유효한지 확인해주세요.")
            else:
                print(f"❌ Whisper CLI 오류: {error_message}")
            return {}

        if not os.path.exists(output_txt_path):
            print(f"❌ Whisper 변환 후 출력 파일({output_txt_path})을 찾을 수 없습니다.")
            return {}

        # 결과 파싱
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read().strip()
        
        speakers = []
        import re
        # 예: [00:00:00.964 --> 00:00:02.484] SPEAKER_00: 안녕하세요.
        pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s(SPEAKER_\d{2}):\s(.*)")
        for line in full_text.split('\n'):
            match = pattern.match(line)
            if match:
                start, end, speaker, text = match.groups()
                speakers.append({
                    "speaker": speaker,
                    "text": text.strip(),
                    "start": start,
                    "end": end
                })

        return {"speakers": speakers, "full_text": full_text}

    except Exception as e:
        print(f"❌ 화자 분리 변환 중 오류: {e}")
        return {}
    finally:
        # 원본 오디오 파일과 생성된 txt 파일 모두 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(output_txt_path):
            os.remove(output_txt_path)


class WhisperLiveStreaming:
    """
    WhisperLiveKit을 사용한 실시간 음성 스트리밍 클래스.
    """
    def __init__(self, model: str = "base", language: str = "ko", diarization: bool = False):
        self.stt_instance = WhisperLiveSTT(model=model, language=language, diarization=diarization)
    
    async def start_streaming(self):
        """
        실시간 음성 스트리밍을 시작합니다.
        """
        print("=== WhisperLiveKit 실시간 스트리밍 시작 ===")
        print(f"모델: {self.stt_instance.model}, 언어: {self.stt_instance.language}, 화자 구분: {'활성화' if self.stt_instance.diarization else '비활성화'}")
        
        if not await self.stt_instance.start_server():
            print("❌ 스트리밍을 시작할 수 없습니다.")
            return
        
        print("\n🛑 Ctrl+C로 서버를 종료할 수 있습니다.\n")
        
        try:
            while self.stt_instance.server_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 스트리밍을 종료합니다...")
        finally:
            await self.stt_instance.stop_server()
