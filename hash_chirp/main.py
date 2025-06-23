import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque
import hashlib
import time

fs = 44100
chunk_size = 1024

print("📡 Gitarg Ambient Blender with Hash Filtering, Chirp Matching, and Volume Decay Activated. Ctrl+C to stop.")

blend_duration_range = (8, 15)
freq_range = (100, 1000)
sweep_step = 1
initial_volume = 0.02
volume_decay = 0.99

current_volume = initial_volume
current_freq = freq_range[0]
direction = 1

mod_index = 0
mod_timer = 0
mod_step_time = 0.01
sweep_cycle_start = time.time()
sweep_duration = np.random.uniform(*blend_duration_range)

high_freq_cutoff = 3000
freq_bin_history = deque(maxlen=1)
hash_history = deque(maxlen=20)
prev_magnitudes = None
last_chirp_time = 0
chirp_debounce = 3


def cancel_all_but_filter_high(audio):
    Y = rfft(audio)
    freqs = rfftfreq(len(audio), 1/fs)
    high_bins = set(np.where(freqs > high_freq_cutoff)[0])
    freq_bin_history.append(high_bins)
    cancel_bins = set.intersection(*freq_bin_history) if freq_bin_history else set()
    for i in cancel_bins:
        Y[i] = 0
    return irfft(-Y).astype(np.float32)


def generate_chirp(f0=3000, f1=1000):
    chirp_t = np.arange(chunk_size) / fs
    chirp_tone = 0.1 * chirp(chirp_t, f0=f0, f1=f1, t1=chirp_t[-1], method='linear')
    return chirp_tone.astype(np.float32)


def callback(indata, outdata, frames, time_info, status):
    global current_freq, direction, sweep_cycle_start, sweep_duration, current_volume, prev_magnitudes, last_chirp_time

    now = time.time()
    elapsed = now - sweep_cycle_start

    if elapsed >= sweep_duration:
        sweep_cycle_start = now
        sweep_duration = np.random.uniform(*blend_duration_range)
        direction *= -1
        current_freq += np.random.choice([-1, 1]) * sweep_step
        current_freq = np.clip(current_freq, *freq_range)

    current_volume *= volume_decay
    current_volume = max(current_volume, 0.0001)

    tone = current_volume * np.sin(2 * np.pi * current_freq * np.arange(frames) / fs).astype(np.float32)

    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)

    hash_digest = hashlib.sha256(input_audio.tobytes()).hexdigest()
    hash_matched = False
    if hash_digest in hash_history:
        input_audio[:] = 0
        hash_matched = True
    else:
        hash_history.append(hash_digest)

    Y = rfft(input_audio)
    freqs = rfftfreq(len(input_audio), 1/fs)
    magnitudes = np.abs(Y)

    chirp_signal = np.zeros(frames, dtype=np.float32)
    if prev_magnitudes is not None and hash_matched and (now - last_chirp_time) > chirp_debounce:
        diff = np.abs(prev_magnitudes - magnitudes)
        if np.any(diff > 0.01):
            sorted_indices = np.argsort(magnitudes)[::-1]
            if len(sorted_indices) > 1:
                f0 = freqs[sorted_indices[0]]
                f1 = freqs[sorted_indices[1]]
                chirp_signal = generate_chirp(f0=f0, f1=f1)
                last_chirp_time = now

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

