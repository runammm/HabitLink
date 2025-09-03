from audio_analysis import *

if __name__ == "__main__":
    filepath = "audios/EMS2_L.wav"
    ap = AudioProcessor(filepath)
    ap.run_full_analysis()
