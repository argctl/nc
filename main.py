import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft
from collections import deque
import zlib

fs = 44100
chunk_size = 1024
input_threshold = 0.03       # Trigger threshold for suppression
recovery_rate = 0.01         # Gain recovery rate per chunk
suppression_gain = 0.2       # Minimum output gain during suppression
current_gain = 1.0

hash_buffer = deque(maxlen=10)
delay_buffer = deque(maxlen=2)  # Small delay to help cancel feedback

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def hash_audio_chunk(chunk):
    normed = np.int16(chunk.flatten() * 32767)
    return zlib.adler32(normed.tobytes())

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

    # Delay buffer to avoid tight loops
    delay_buffer.append(flipped)
    if len(delay_buffer) < delay_buffer.maxlen:
        outdata[:] = np.zeros_like(indata)
        return

    delayed_flipped = delay_buffer[0]

    # Hash detection
    h = hash_audio_chunk(delayed_flipped)
    if h in hash_buffer:
        delayed_flipped *= 0.1  # Suppress repeated patterns
    else:
        hash_buffer.append(h)

    # Adaptive gain logic
    if input_vol > input_threshold:
        current_gain = suppression_gain
    else:
        current_gain = min(1.0, current_gain + recovery_rate)

    outdata[:] = delayed_flipped * current_gain

print("🔉 Gitarg: Hash + Delay + Counter-Adaptive Gain Active. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("⏹️ Gitarg stopped.")
