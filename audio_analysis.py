import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AudioProcessor:
    def __init__(self, filepath):
        # 1) Librosa로 오디오 로드 (mono, 원본 sr 유지)
        self.data, self.fs = librosa.load(filepath, sr=None, mono=True)
        self.duration = len(self.data) / self.fs

    def bandpass(self, lowcut, highcut):
        """
        FFT 기반 band-pass filtering (Non-casual)
        """
        x = self.data
        N = len(x)
        # 1) FFT
        X = np.fft.rfft(x)
        # 2) 주파수 축
        freqs = np.fft.rfftfreq(N, d=1/self.fs)
        # 3) 마스크 생성
        mask = (freqs >= lowcut) & (freqs <= highcut)
        # 4) 대역 통과
        X_filt = X * mask
        # 5) 역변환
        return np.fft.irfft(X_filt, n=N)

    def compute_stft(self, n_fft=1024, hop_length=None):
        """
        Librosa STFT → (freqs, times, complex_spectrogram)
        """
        if hop_length is None:
            hop_length = n_fft // 2
        D = librosa.stft(self.data, n_fft=n_fft, hop_length=hop_length)
        freqs = np.linspace(0, self.fs/2, 1 + n_fft//2)
        times = np.arange(D.shape[1]) * hop_length / self.fs
        return freqs, times, D

    def compute_spectrogram(self, n_fft=1024, hop_length=None):
        """
        STFT → dB 스펙트로그램
        """
        freqs, times, D = self.compute_stft(n_fft, hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return freqs, times, S_db

    def compute_psd(self, data=None):
        """
        Periodogram 방식 PSD 계산
        Pxx = (1/(fs*N)) * |FFT(x)|^2
        """
        x = self.data if data is None else data
        N = len(x)
        X = np.fft.rfft(x)
        Pxx = (1.0 / (self.fs * N)) * (np.abs(X) ** 2)
        freqs = np.fft.rfftfreq(N, d=1/self.fs)
        return freqs, Pxx

    def detect_peaks(self, spectrum, height=None):
        """
        1D 스펙트럼에서 단순 피크 검출
        """
        peaks = []
        for i in range(1, len(spectrum)-1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if height is None or spectrum[i] >= height:
                    peaks.append(i)
        return np.array(peaks)

    def convergence_check(self, feature_sequence, tol=1e-3, window=5):
        """
        마지막 window 개수 내에 tol 이하 변화 → 수렴 판단
        """
        seq = np.asarray(feature_sequence)
        if len(seq) < window:
            return False
        last = seq[-window:]
        return np.max(last) - np.min(last) < tol

    def plot_results(self, freqs, times, S_db):
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, S_db, shading='gouraud')
        plt.title("Spectrogram (dB)")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        # 1) Band-pass (Ex. 음성 대역 300-3400 Hz)
        filtered = self.bandpass(300, 3400)

        # 2) PSD 계산
        f_psd, Pxx = self.compute_psd(filtered)

        # 3) Peak 검출 (Ex. PSD 값 중 상위 5개 피크)
        peaks = self.detect_peaks(Pxx, height=None)
        top5 = peaks[np.argsort(Pxx[peaks])[-5:]][::-1]

        # 4) 수렴 여부 판단
        is_conv = self.convergence_check(Pxx, tol=1e-6, window=10)

        # 5) 콘솔 출력
        print(f"Audio duration   : {self.duration:.2f} sec")
        print(f"PSD converged?   : {is_conv}")
        print("Top 5 PSD peaks  :")
        for idx in top5:
            print(f"  freq={f_psd[idx]:.1f} Hz, PSD={Pxx[idx]:.3e}")

        # 6) 스펙트로그램 플롯
        f_sp, t_sp, S_db = self.compute_spectrogram()
        self.plot_results(f_sp, t_sp, S_db)
