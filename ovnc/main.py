import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft

fs = 44100
chunk_size = 1024
input_threshold = 0.03       # Suppression trigger threshold (tune as needed)
recovery_rate = 0.01         # How quickly gain recovers (per chunk)
suppression_gain = 0.2       # Minimum output gain when input is loud

current_gain = 1.0           # Output gain, dynamic

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def flip_phase_all_freq(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

def callback(indata, outdata, frames, time, status):
    global current_gain

    if status:
        print("⚠️", status)

    input_vol = rms(indata)
    flipped = flip_phase_all_freq(indata)

    if input_vol > input_threshold:
        current_gain = suppression_gain  # Suppress immediately
    else:
        current_gain = min(1.0, current_gain + recovery_rate)  # Recover slowly

    outdata[:] = flipped * current_gain

print("🔉 Gitarg: Counter-adaptive volume control active. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("⏹️ Gitarg stopped.")
