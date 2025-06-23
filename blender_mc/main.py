import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque
import hashlib
import time

fs = 44100
chunk_size = 1024

print("📡 Gitarg Ambient Blender with Echo Hash Filtering, Chirp Detection, and Volume Decay Activated. Ctrl+C to stop.")

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
freq_bin_history = deque(maxlen=1)

# Cache hashes for recent input to detect repeated reverb/echo
hash_history = deque(maxlen=20)

# Track frequency magnitudes to detect matching peaks
prev_magnitudes = None

# FFT cancellation with high-frequency filtering and hash-based echo rejection
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

def generate_chirp():
    chirp_t = np.arange(chunk_size) / fs
    chirp_tone = 0.1 * chirp(chirp_t, f0=1000, f1=3000, t1=chirp_t[-1], method='linear')
    return chirp_tone.astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global current_freq, direction, sweep_cycle_start, sweep_duration, current_volume, prev_magnitudes

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

    # Hash-based filtering to suppress echo/reverb
    hash_digest = hashlib.sha256(input_audio.tobytes()).hexdigest()
    if hash_digest in hash_history:
        input_audio[:] = 0
    else:
        hash_history.append(hash_digest)

    # Frequency domain analysis
    Y = rfft(input_audio)
    magnitudes = np.abs(Y)

    chirp_signal = np.zeros(frames, dtype=np.float32)
    if prev_magnitudes is not None:
        matching_bins = np.where(np.isclose(prev_magnitudes, magnitudes, atol=1e-2))[0]
        if len(matching_bins) > 0:
            chirp_signal = generate_chirp()

    prev_magnitudes = magnitudes

    cancellation = cancel_all_but_filter_high(input_audio)
    mix = tone + cancellation[:frames] + chirp_signal[:frames]
    outdata[:, 0] = mix

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Ambient Blender Stopped.")
