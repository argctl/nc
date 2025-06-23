import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque
import time

fs = 44100
chunk_size = 1024
high_freq_cutoff = 3000

print("📡 Gitarg Ambient Blender Activated. Ctrl+C to stop.")

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


def generate_tone(freq, duration_sec):
    samples = np.arange(int(fs * duration_sec)) / fs
    tone = volume * np.sin(2 * np.pi * freq * samples)
    return tone.astype(np.float32)


def callback(indata, outdata, frames, time_info, status):
    global current_freq, direction, sweep_cycle_start, sweep_duration

    now = time.time()
    elapsed = now - sweep_cycle_start

    if elapsed >= sweep_duration:
        sweep_cycle_start = now
        sweep_duration = np.random.uniform(*blend_duration_range)
        direction *= -1  # Reverse sweep
        current_freq += np.random.choice([-1, 1]) * sweep_step  # Drift
        current_freq = np.clip(current_freq, *freq_range)

    tone = volume * np.sin(2 * np.pi * current_freq * np.arange(frames) / fs).astype(np.float32)
    outdata[:] = tone.reshape(-1, 1)

try:
    with sd.OutputStream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Ambient Blender Stopped.")

