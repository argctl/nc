import sounddevice as sd
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

fs = 44100
chunk_size = 1024

print("🔁 Gitarg: Real-time Phase Flip (Frequency-Domain) Started. Ctrl+C to stop")

def flip_phase_all_freq(audio_chunk):
    # Flatten and FFT
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y  # Flip phase of all frequencies
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

def callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)

    # Apply frequency-domain phase flip
    flipped = flip_phase_all_freq(indata)
    outdata[:] = flipped

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg stream stopped.")
