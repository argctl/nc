import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft
from collections import deque

fs = 44100
chunk_size = 1024
delay_chunks = 5  # Delay window for RMS comparison
input_rms_history = deque(maxlen=delay_chunks)

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def flip_phase_all_freq(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

def callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)

    # Calculate RMS of current input
    current_input_rms = rms(indata)
    input_rms_history.append(current_input_rms)

    # Apply frequency domain phase flip
    flipped = flip_phase_all_freq(indata)

    # Compute RMS of current output
    output_rms = rms(flipped)

    # Compare against delayed input RMS
    if len(input_rms_history) == input_rms_history.maxlen:
        max_rms = input_rms_history[0] + 1e-6  # Avoid div by zero
        scale = min(1.0, max_rms / (output_rms + 1e-6))
        outdata[:] = flipped * scale
    else:
        outdata[:] = flipped  # No scaling until buffer fills

print("🔊 Gitarg: Output capped to delayed input RMS. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("⏹️ Stopped.")
