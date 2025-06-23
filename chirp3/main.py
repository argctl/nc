import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque

fs = 44100
chunk_size = 1024
min_spike_threshold = 0.01
low_freq_cutoff = 500  # Focus below this frequency (Hz)
min_duration_sec = 1.0  # Frequency must persist for at least this long to be removed

# Chirp generation config
def generate_chirp(duration_ms, direction='up'):
    duration_sec = duration_ms / 1000.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    if direction == 'up':
        wave = chirp(t, f0=300, f1=3000, t1=duration_sec, method='linear')
    else:
        wave = chirp(t, f0=3000, f1=300, t1=duration_sec, method='linear')
    return wave.astype(np.float32).reshape(-1, 1)

# Frequency tracker for persistent tones
freq_persistence = {}
freq_history = deque(maxlen=int((fs / chunk_size) * min_duration_sec))

# Remove frequencies that persist for longer than threshold
def cancel_persistent_freqs(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    freqs = rfftfreq(len(y), 1/fs)
    mag = np.abs(Y)

    current = set()
    threshold = np.percentile(mag, 95) * 0.5  # Adaptive threshold
    for i, m in enumerate(mag):
        if m > threshold:
            current.add(i)

    freq_history.append(current)

    # Count how often each bin appears
    freq_count = {}
    for history in freq_history:
        for bin_index in history:
            freq_count[bin_index] = freq_count.get(bin_index, 0) + 1

    # Zero out persistent bins
    for bin_index, count in freq_count.items():
        if count >= len(freq_history):
            Y[bin_index] = 0

    y_out = irfft(Y)
    return y_out.reshape((-1, 1))

# Runtime state
last_peak_rms = 0.0
chirp_active = False
chirp_data = np.zeros((0, 1), dtype=np.float32)
silence_remaining = 0

print("📡 Gitarg Pulse-Chirp with Persistent Frequency Cancellation Activated. Ctrl+C to stop.")

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def callback(indata, outdata, frames, time, status):
    global last_peak_rms, chirp_active, chirp_data, silence_remaining

    if status:
        print("⚠️", status)

    # Cancel persistent frequencies
    cleaned = cancel_persistent_freqs(indata)

    # Monitor input RMS
    current_rms = rms(indata)
    delta = current_rms - last_peak_rms

    # Check for upward spike in quiet noise
    if not chirp_active and delta > min_spike_threshold:
        chirp_duration_ms = max(1, int(delta * 1000))  # Duration proportional to spike
        chirp_wave = generate_chirp(chirp_duration_ms, direction='up' if np.random.rand() > 0.5 else 'down')
        chirp_data = chirp_wave
        silence_remaining = len(chirp_wave)
        chirp_active = True
        last_peak_rms = current_rms

    # Output audio
    if chirp_active and len(chirp_data) > 0:
        out_len = min(len(chirp_data), frames)
        outdata[:out_len] = chirp_data[:out_len]
        if out_len < frames:
            outdata[out_len:] = cleaned[out_len:]
        chirp_data = chirp_data[out_len:]
        if len(chirp_data) == 0:
            chirp_active = False
    elif silence_remaining > 0:
        outdata[:] = 0
        silence_remaining -= frames
    else:
        outdata[:] = cleaned

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Pulse-Chirp with Cancellation Stopped.")

