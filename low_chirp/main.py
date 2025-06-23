import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft, rfftfreq
from collections import deque

fs = 44100
chunk_size = 1024
min_spike_threshold = 0.01
high_freq_cutoff = 3000  # Remove frequencies above this threshold (Hz)
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
freq_history = deque(maxlen=int((fs / chunk_size) * min_duration_sec))

# Remove high frequencies and persistent frequencies, also detect dominant low frequencies
def cancel_and_extract_low_freqs(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    freqs = rfftfreq(len(y), 1/fs)
    mag = np.abs(Y)

    current = set()
    threshold = np.percentile(mag, 95) * 0.5
    for i, (f, m) in enumerate(zip(freqs, mag)):
        if m > threshold and f <= high_freq_cutoff:
            current.add(i)

    freq_history.append(current)

    # Track frequency persistence
    freq_count = {}
    for history in freq_history:
        for bin_index in history:
            freq_count[bin_index] = freq_count.get(bin_index, 0) + 1

    # Cancel persistent and high frequencies
    for i, f in enumerate(freqs):
        if f > high_freq_cutoff or freq_count.get(i, 0) >= len(freq_history):
            Y[i] = 0

    # Detect strongest low frequency
    low_freq_mask = freqs < 300  # Define low frequency range
    low_mag = mag * low_freq_mask
    dominant_index = np.argmax(low_mag)
    inverted_low = np.zeros_like(Y)
    inverted_low[dominant_index] = -Y[dominant_index] * 2  # Strong inversion

    # Create cancellation stream
    cancellation_wave = irfft(inverted_low).reshape((-1, 1))

    y_out = irfft(Y).reshape((-1, 1))
    return y_out, cancellation_wave

# Runtime state
last_peak_rms = 0.0
chirp_active = False
chirp_data = np.zeros((0, 1), dtype=np.float32)
silence_remaining = 0

print("📡 Gitarg Pulse-Chirp with Enhanced Low-Freq Cancellation Activated. Ctrl+C to stop.")

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def callback(indata, outdata, frames, time, status):
    global last_peak_rms, chirp_active, chirp_data, silence_remaining

    if status:
        print("⚠️", status)

    # Process audio
    cleaned, cancellation = cancel_and_extract_low_freqs(indata)

    # Monitor input RMS
    current_rms = rms(indata)
    delta = current_rms - last_peak_rms

    if not chirp_active and delta > min_spike_threshold:
        chirp_duration_ms = max(1, int(delta * 1000))
        chirp_wave = generate_chirp(chirp_duration_ms, direction='up' if np.random.rand() > 0.5 else 'down')
        chirp_data = chirp_wave
        silence_remaining = len(chirp_wave)
        chirp_active = True
        last_peak_rms = current_rms

    # Combine outputs
    final_out = cleaned + cancellation

    if chirp_active and len(chirp_data) > 0:
        out_len = min(len(chirp_data), frames)
        final_out[:out_len] = chirp_data[:out_len]
        if out_len < frames:
            final_out[out_len:] = cleaned[out_len:] + cancellation[out_len:]
        chirp_data = chirp_data[out_len:]
        if len(chirp_data) == 0:
            chirp_active = False
    elif silence_remaining > 0:
        final_out[:] = 0
        silence_remaining -= frames

    outdata[:] = final_out

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Pulse-Chirp with Cancellation Stopped.")

