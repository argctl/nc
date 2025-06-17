import numpy as np
import sounddevice as sd
import zlib
from collections import deque
from scipy.fft import rfft, irfft

fs = 44100
chunk_size = 1024
hash_buffer = deque(maxlen=10)  # Store recent hashes

def hash_audio_chunk(chunk):
    normed = np.int16(chunk.flatten() * 32767)  # Convert to PCM-like format
    return zlib.adler32(normed.tobytes())

def flip_phase_all_freq(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

def callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)

    flipped = flip_phase_all_freq(indata)

    # Hash the flipped chunk
    h = hash_audio_chunk(flipped)

    if h in hash_buffer:
        # Repeated chunk — suppress or attenuate output
        outdata[:] = 0.1 * flipped  # Or np.zeros_like(flipped)
    else:
        outdata[:] = flipped
        hash_buffer.append(h)

print("🎛️ Gitarg running with hash-based feedback reduction. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg stopped.")
