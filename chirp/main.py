import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, irfft

fs = 44100
chunk_size = 1024
min_spike_threshold = 0.01

# Chirp generation config
def generate_chirp(duration_ms, direction='up'):
    duration_sec = duration_ms / 1000.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    if direction == 'up':
        wave = chirp(t, f0=300, f1=3000, t1=duration_sec, method='linear')
    else:
        wave = chirp(t, f0=3000, f1=300, t1=duration_sec, method='linear')
    return wave.astype(np.float32).reshape(-1, 1)

# Noise cancellation via phase inversion
def flip_phase_all_freq(audio_chunk):
    y = audio_chunk.flatten()
    Y = rfft(y)
    Y_flipped = -Y
    y_out = irfft(Y_flipped)
    return y_out.reshape((-1, 1))

# Runtime state
last_peak_rms = 0.0
chirp_active = False
chirp_data = np.zeros((0, 1), dtype=np.float32)
silence_remaining = 0

print("📡 Gitarg Pulse-Chirp with Noise Cancellation Activated. Ctrl+C to stop.")

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def callback(indata, outdata, frames, time, status):
    global last_peak_rms, chirp_active, chirp_data, silence_remaining

    if status:
        print("⚠️", status)

    # Flip phase for noise cancellation
    flipped = flip_phase_all_freq(indata)

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
            outdata[out_len:] = flipped[out_len:]
        chirp_data = chirp_data[out_len:]
        if len(chirp_data) == 0:
            chirp_active = False
    elif silence_remaining > 0:
        outdata[:] = 0
        silence_remaining -= frames
    else:
        outdata[:] = flipped

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Pulse-Chirp with Cancellation Stopped.")

