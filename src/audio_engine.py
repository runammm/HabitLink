import numpy as np
import sounddevice as sd
import asyncio
import concurrent.futures

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


class AudioEngine:
    def __init__(self, samplerate: int = 16000, channels: int = 1, dtype: str = "float32"):
        """
        사운드 디바이스 라이브러리를 활용한 오디오 녹음 엔진
        
        Args:
            samplerate (int): 샘플링 레이트 (기본값: 16000Hz)
            channels (int): 채널 수 (기본값: 1, 모노)
            dtype (str): 데이터 타입 (기본값: "float32")
        """
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.dtype = dtype

    def record(self, duration: float, output_path: str = "temp.wav") -> str:
        """
        주어진 시간(duration)만큼 마이크에서 녹음하여 파일로 저장합니다.
        
        Args:
            duration (float): 녹음 시간 (초)
            output_path (str): 저장할 파일 경로 (기본값: "temp.wav")
            
        Returns:
            str: 저장된 파일 경로
            
        Raises:
            ValueError: duration이 0 이하일 때
            RuntimeError: soundfile이 설치되지 않았을 때
        """
        if duration <= 0:
            raise ValueError("duration은 0보다 커야 합니다.")
            
        if not _HAS_SOUNDFILE:
            raise RuntimeError("파일로 저장하려면 soundfile이 필요합니다. 'pip install soundfile'로 설치하세요.")
        
        # 총 프레임 수 계산
        frames = int(round(self.samplerate * duration))
        
        print(f"녹음 시작... ({duration}초)")
        
        # 녹음 실행
        recording = sd.rec(
            frames=frames,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        
        # 녹음 완료까지 대기
        sd.wait()
        
        print("녹음 완료!")
        
        # 파일로 저장
        sf.write(output_path, recording, self.samplerate)
        
        print(f"파일 저장됨: {output_path}")
        
        return output_path

    async def record_async(self, duration: float, output_path: str = "temp.wav") -> str:
        """
        비동기로 주어진 시간(duration)만큼 마이크에서 녹음하여 파일로 저장합니다.
        
        Args:
            duration (float): 녹음 시간 (초)
            output_path (str): 저장할 파일 경로 (기본값: "temp.wav")
            
        Returns:
            str: 저장된 파일 경로
            
        Raises:
            ValueError: duration이 0 이하일 때
            RuntimeError: soundfile이 설치되지 않았을 때
        """
        if duration <= 0:
            raise ValueError("duration은 0보다 커야 합니다.")
            
        if not _HAS_SOUNDFILE:
            raise RuntimeError("파일로 저장하려면 soundfile이 필요합니다. 'pip install soundfile'로 설치하세요.")
        
        # 총 프레임 수 계산
        frames = int(round(self.samplerate * duration))
        
        print(f"비동기 녹음 시작... ({duration}초)")
        
        # 스레드 풀에서 녹음 실행
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 녹음 작업을 별도 스레드에서 실행
            recording = await loop.run_in_executor(
                executor, 
                self._record_blocking, 
                frames
            )
        
        print("비동기 녹음 완료!")
        
        # 파일로 저장
        await loop.run_in_executor(
            executor,
            sf.write,
            output_path,
            recording,
            self.samplerate
        )
        
        print(f"파일 저장됨: {output_path}")
        
        return output_path

    def _record_blocking(self, frames: int) -> np.ndarray:
        """
        동기적으로 녹음을 수행하는 내부 메서드
        
        Args:
            frames (int): 녹음할 프레임 수
            
        Returns:
            np.ndarray: 녹음된 오디오 데이터
        """
        recording = sd.rec(
            frames=frames,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        
        # 녹음 완료까지 대기
        sd.wait()
        
        return recording
        
    def get_device_info(self):
        """
        사용 가능한 오디오 디바이스 정보를 반환합니다.
        
        Returns:
            dict: 디바이스 정보
        """
        return {
            "devices": sd.query_devices(),
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1]
        }


# 비동기 사용 예제
async def main():
    audio_engine = AudioEngine()
    
    # 비동기 녹음
    result = await audio_engine.record_async(5.0, "async_recording.wav")
    print(f"비동기 녹음 결과: {result}")


if __name__ == "__main__":
    # 동기 녹음 테스트
    audio_engine = AudioEngine()
    audio_engine.record(5.0)
    
    # 비동기 녹음 테스트
    # asyncio.run(main())
    