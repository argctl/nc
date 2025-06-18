# --- Script 1: Volume Upward Clamp and Reduction ---

import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft

fs = 44100
chunk_size = 1024
max_rms_threshold = 0.05   # Upper RMS limit
reduction_factor = 0.7     # Reduce volume to 70% when over threshold

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

    flipped = flip_phase_all_freq(indata)
    volume = rms(flipped)

    if volume > max_rms_threshold:
        flipped *= reduction_factor

    outdata[:] = flipped

print("🔊 Gitarg Clamp: Volume capped and reduced on peak. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("⏹️ Gitarg Clamp stopped.")


# --- Script 2: Oscillating Volume Modulation with Expanding Cycles ---

import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft

fs = 44100
chunk_size = 1024
base_cycle = 20  # number of chunks
cycle_increase = 10
max_gain = 1.0
min_gain = 0.3

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def flip_phase_all_freq(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

# State for oscillating gain
cycle_count = 0
current_cycle = base_cycle
gain_direction = 1  # 1 = increasing, -1 = decreasing
gain = min_gain
next_target_gain = max_gain

def callback(indata, outdata, frames, time, status):
    global cycle_count, current_cycle, gain_direction, gain, next_target_gain

    if status:
        print("⚠️", status)

    flipped = flip_phase_all_freq(indata)

    # Linear gain transition within cycle
    step = (next_target_gain - gain) / (current_cycle - cycle_count + 1)
    gain += step
    outdata[:] = flipped * gain

    # Advance cycle count
    cycle_count += 1

    if cycle_count >= current_cycle:
        cycle_count = 0
        current_cycle += cycle_increase
        gain_direction *= -1

        if gain_direction > 0:
            # Next target: slightly lower max
            next_target_gain = max(min_gain, next_target_gain - 0.1)
        else:
            next_target_gain = max_gain

print("🔄 Gitarg Oscillator: Dynamic volume cycling. Ctrl+C to stop.")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("⏹️ Gitarg Oscillator stopped.")
