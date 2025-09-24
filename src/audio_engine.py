import numpy as np
import sounddevice as sd
import asyncio
import concurrent.futures
import soundfile as sf


class AudioEngine:
    def __init__(self, samplerate: int = 16000, channels: int = 1, dtype: str = "float32"):
        """
        Audio recording engine utilizing the sounddevice library
        
        Args:
            samplerate (int): Sampling rate (Default: 16000Hz)
            channels (int): Number of channels (Default: 1, Mono)
            dtype (str): Data type (Default: "float32")
        """
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.dtype = dtype

    def record(self, duration: float, output_path: str = "temp.wav") -> str:
        """
        Record from the microphone for a given duration and save it as a file.
        
        Args:
            duration (float): Recording time (seconds)
            output_path (str): Path to save file (Default: "temp.wav")
            
        Returns:
            str: Saved file path
            
        Raise:
            ValueError: When duration is less than or equal to 0
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than 0.")
        
        # Calculate total frame count
        frames = int(round(self.samplerate * duration))
        
        print(f"Start recording... ({duration} seconds)")
        
        # Run recording
        recording = sd.rec(
            frames=frames,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        
        # Wait until recording is complete
        sd.wait()
        
        print("Recording complete!")
        
        # Save to file
        sf.write(output_path, recording, self.samplerate)
        
        print(f"File saved: {output_path}")
        
        return output_path

    async def record_async(self, duration: float, output_path: str = "temp.wav") -> str:
        """
        Asynchronously records from the microphone for a given duration and saves it as a file.
        
        Args:
            duration (float): Recording time (seconds)
            output_path (str): Path to save file (Default: "temp.wav")
            
        Returns:
            str: Saved file path
            
        Raise:
            ValueError: When duration is less than or equal to 0
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than 0.")
        
        # Calculate total frame count
        frames = int(round(self.samplerate * duration))
        
        print(f"Starting asynchronous recording... ({duration} seconds)")
        
        # Running recordings on a thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run the recording task in a separate thread
            recording = await loop.run_in_executor(
                executor, 
                self._record_blocking, 
                frames
            )
        
        print("Asynchronous recording complete!")
        
        # Save to file
        await loop.run_in_executor(
            executor,
            sf.write,
            output_path,
            recording,
            self.samplerate
        )
        
        print(f"File saved: {output_path}")
        
        return output_path

    def _record_blocking(self, frames: int) -> np.ndarray:
        """
        Internal method that performs recording synchronously
        
        Args:
            frames (int): Number of frames to record
            
        Returns:
            np.ndarray: Recorded audio data
        """
        recording = sd.rec(
            frames=frames,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        
        # Wait until recording is complete
        sd.wait()
        
        return recording

    def get_device_info(self):
        """
        Returns information about available audio devices.
        
        Returns:
            dict: Device information
        """
        return {
            "devices": sd.query_devices(),
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1]
        }


# Test code
# Async usage example
async def main():
    audio_engine = AudioEngine()
    
    # Async recording
    result = await audio_engine.record_async(5.0, "async_recording.wav")
    print(f"Asynchronous recording result: {result}")


if __name__ == "__main__":
    # Synchronous recording test
    audio_engine = AudioEngine()
    audio_engine.record(5.0)
    
    # Asynchronous recording test
    # asyncio.run(main())
    