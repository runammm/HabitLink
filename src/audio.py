import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False


class AudioInterface:
    """
    비동기 오디오 녹음 인터페이스.

    사용 예:
        ai = AudioInterface(samplerate=16000, channels=1, dtype='float32')
        data = await ai.record(5.0)  # 5초 비동기 녹음
        await ai.record_to_file(3.0, "output.wav")  # 3초 녹음 후 파일 저장
    """
    def __init__(self, samplerate: int = 16000, channels: int = 1, dtype: str = "float32"):
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.dtype = dtype

    async def record(self, seconds: float) -> np.ndarray:
        """
        주어진 시간(seconds)만큼 마이크에서 녹음하여 numpy 배열을 반환합니다. (비동기)
        반환 shape: (frames, channels) 또는 channels=1이면 (frames, 1)
        """
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("seconds는 0보다 커야 합니다.")
        frames = int(round(self.samplerate * seconds))

        def _blocking_record() -> np.ndarray:
            data = sd.rec(frames, samplerate=self.samplerate, channels=self.channels, dtype=self.dtype)
            sd.wait()
            return np.asarray(data).copy()

        return await asyncio.to_thread(_blocking_record)

    async def record_to_file(self, seconds: float, filepath: str, subtype: Optional[str] = "PCM_16") -> str:
        """
        주어진 시간(seconds)만큼 녹음 후 파일로 저장합니다. (비동기)
        파일 포맷은 filepath 확장자에 따릅니다. (예: .wav, .flac 등; soundfile 필요)
        """
        if not _HAS_SF:
            raise RuntimeError("파일로 저장하려면 soundfile이 필요합니다. pip install soundfile 후 다시 시도하세요.")
        data = await self.record(seconds)

        def _write():
            sf.write(filepath, data, self.samplerate, subtype=subtype)

        await asyncio.to_thread(_write)
        return filepath
