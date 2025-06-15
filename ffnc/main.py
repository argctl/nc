import sounddevice as sd
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import fftconvolve
import noisereduce as nr

fs = 44100                # Sampling rate (Hz)
chunk_duration = 2        # Chunk duration (seconds)

# Frequencies to filter and corresponding bandwidths
noise_freqs = [60, 120, 180, 240, 300]  # Common noise frequencies (e.g., hum)
bandwidth = 2                           # Hz bandwidth around each noise frequency

# Noise gate parameters
gate_threshold = 0.02
attack_release_duration = 0.01

# Echo reduction parameters
echo_decay = 0.6
echo_delay = 0.02  # 20 ms typical echo

# Frequency domain filter function
def freq_filter(audio_chunk, fs, noise_freqs, bandwidth):
    N = len(audio_chunk)
    yf = rfft(audio_chunk)
    xf = rfftfreq(N, 1/fs)

    mask = np.ones_like(yf)
    for f_noise in noise_freqs:
        mask[(xf > f_noise - bandwidth) & (xf < f_noise + bandwidth)] = 0

    filtered_yf = yf * mask
    return irfft(filtered_yf, n=N)

# Noise gate function
def noise_gate(audio, threshold):
    envelope = np.abs(audio)
    gate = envelope > threshold
    smooth_gate = np.convolve(gate, np.ones(int(fs * attack_release_duration)) / int(fs * attack_release_duration), mode='same')
    return audio * smooth_gate

# Echo reduction function
def reduce_echo(audio, fs, decay, delay):
    delay_samples = int(delay * fs)
    impulse_response = np.zeros(delay_samples * 2)
    impulse_response[0] = 1
    impulse_response[delay_samples] = -decay
    return fftconvolve(audio, impulse_response, mode='same')

print("🚀 Real-time combined frequency, echo, and noise gating started. Press Ctrl+C to stop.")

# Capture initial ambient noise profile
print("Recording initial ambient noise profile...")
noise_sample = sd.rec(int(fs * 3), samplerate=fs, channels=1, dtype='float32').flatten()
sd.wait()

try:
    while True:
        audio_chunk = sd.rec(int(fs * chunk_duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_chunk = audio_chunk.flatten()

        # Frequency-based noise filtering
        audio_filtered_freq = freq_filter(audio_chunk, fs, noise_freqs, bandwidth)

        # Noise gating
        audio_gated = noise_gate(audio_filtered_freq, gate_threshold)

        # Echo reduction
        audio_deecho = reduce_echo(audio_gated, fs, echo_decay, echo_delay)

        # Additional noise reduction using ambient noise profile
        audio_final = nr.reduce_noise(y=audio_deecho, y_noise=noise_sample, sr=fs)

        # Playback
        sd.play(audio_final, fs)
        sd.wait()

except KeyboardInterrupt:
    print("\n⏹️ Stopped real-time audio processing.")

