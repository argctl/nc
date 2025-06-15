import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy.signal import fftconvolve

fs = 44100  # Sampling rate
chunk_duration = 1  # seconds

# Simple noise gate parameters
threshold = 0.02
attack_release_duration = 0.01  # smooth gate transitions

def noise_gate(audio, threshold):
    envelope = np.abs(audio)
    gate = envelope > threshold
    smooth_gate = np.convolve(gate, np.ones(int(fs * attack_release_duration))/int(fs * attack_release_duration), mode='same')
    return audio * smooth_gate

def reduce_echo(audio, fs, decay=0.6, delay=0.02):
    delay_samples = int(delay * fs)
    impulse_response = np.zeros(delay_samples * 2)
    impulse_response[0] = 1
    impulse_response[delay_samples] = -decay
    return fftconvolve(audio, impulse_response, mode='same')

print("🚀 Real-time echo reduction started. Press Ctrl+C to stop.")
try:
    while True:
        audio_chunk = sd.rec(int(fs * chunk_duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_chunk = audio_chunk.flatten()

        # Step 1: Apply noise gate to reduce ambient noise and subtle reverberations
        gated_audio = noise_gate(audio_chunk, threshold)

        # Step 2: Echo Reduction via spectral processing
        deecho_audio = reduce_echo(gated_audio, fs)

        # Optional: Further reduce any lingering noise
        final_audio = nr.reduce_noise(y=deecho_audio, y_noise=deecho_audio[:fs//2], sr=fs)

        sd.play(final_audio, fs)
        sd.wait()

except KeyboardInterrupt:
    print("\n⏹️ Stopped real-time echo reduction.")

