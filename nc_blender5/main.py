import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque
import time

fs = 44100
chunk_size = 1024

print("📡 Gitarg Ambient Blender with High-Freq Frame Noise Filtering and Volume Decay Activated. Ctrl+C to stop.")

# Oscillating tone parameters
blend_duration_range = (8, 15)  # Slower oscillation
freq_range = (100, 1000)  # Range to sweep
sweep_step = 1  # Frequency step
initial_volume = 0.02  # Barely audible
volume_decay = 0.99  # Regular decay multiplier

current_volume = initial_volume

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

# Track high-frequency bins over frames
high_freq_cutoff = 3000
freq_bin_history = deque(maxlen=1)  # Only track current frame

# FFT cancellation with high-frequency filtering
def cancel_all_but_filter_high(audio):
    Y = rfft(audio)
    freqs = rfftfreq(len(audio), 1/fs)

    # Identify high-frequency bins
    high_bins = set(np.where(freqs > high_freq_cutoff)[0])
    freq_bin_history.append(high_bins)

    # Zero out any high frequency present in this frame
    cancel_bins = set.intersection(*freq_bin_history) if freq_bin_history else set()
    for i in cancel_bins:
        Y[i] = 0

    return irfft(-Y).astype(np.float32)

def generate_tone(freq, duration_sec, amp):
    samples = np.arange(int(fs * duration_sec)) / fs
    tone = amp * np.sin(2 * np.pi * freq * samples)
    return tone.astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global current_freq, direction, sweep_cycle_start, sweep_duration, current_volume

    now = time.time()
    elapsed = now - sweep_cycle_start

    if elapsed >= sweep_duration:
        sweep_cycle_start = now
        sweep_duration = np.random.uniform(*blend_duration_range)
        direction *= -1
        current_freq += np.random.choice([-1, 1]) * sweep_step
        current_freq = np.clip(current_freq, *freq_range)

    # Apply volume decay
    current_volume *= volume_decay
    current_volume = max(current_volume, 0.0001)  # Prevent silence

    tone = current_volume * np.sin(2 * np.pi * current_freq * np.arange(frames) / fs).astype(np.float32)

    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)
    cancellation = cancel_all_but_filter_high(input_audio)

    mix = tone + cancellation[:frames]
    outdata[:, 0] = mix

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Ambient Blender Stopped.")

