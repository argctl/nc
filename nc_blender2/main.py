import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque
import time

fs = 44100
chunk_size = 1024

print("📡 Gitarg Ambient Blender with Full-Freq Noise Cancellation Activated. Ctrl+C to stop.")

# Oscillating tone parameters
blend_duration_range = (3, 7)
freq_range = (100, 1000)  # Range to sweep
sweep_step = 1  # Frequency step
volume = 0.02  # Barely audible

start_time = time.time()
current_freq = freq_range[0]
direction = 1  # 1 = up, -1 = down

# Tone generator
blend_buffer = np.zeros((chunk_size,), dtype=np.float32)
t = np.arange(chunk_size) / fs

# Runtime frequency modulation state
mod_index = 0
mod_timer = 0
mod_step_time = 0.01
sweep_cycle_start = time.time()
sweep_duration = np.random.uniform(*blend_duration_range)

# Buffer to keep last chunk for cancellation
last_input = np.zeros((chunk_size,), dtype=np.float32)

def generate_tone(freq, duration_sec):
    samples = np.arange(int(fs * duration_sec)) / fs
    tone = volume * np.sin(2 * np.pi * freq * samples)
    return tone.astype(np.float32)

def cancel_all_frequencies(audio):
    Y = rfft(audio)
    Y_canceled = -Y  # Invert all frequencies
    return irfft(Y_canceled).astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global current_freq, direction, sweep_cycle_start, sweep_duration, last_input

    now = time.time()
    elapsed = now - sweep_cycle_start

    if elapsed >= sweep_duration:
        sweep_cycle_start = now
        sweep_duration = np.random.uniform(*blend_duration_range)
        direction *= -1  # Reverse sweep
        current_freq += np.random.choice([-1, 1]) * sweep_step  # Drift
        current_freq = np.clip(current_freq, *freq_range)

    # Generate ambient blend tone
    tone = volume * np.sin(2 * np.pi * current_freq * np.arange(frames) / fs).astype(np.float32)

    # Full-frequency noise cancellation
    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)
    cancellation = cancel_all_frequencies(input_audio)

    # Mix tone and cancellation
    mix = tone + cancellation[:frames]
    outdata[:, 0] = mix

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Ambient Blender Stopped.")

