import numpy as np
import sounddevice as sd

# Sampling rate and buffer size
fs = 44100
chunk_size = 1024

# ELF tone parameters
freq = 10  # 10 Hz — Extremely Low Frequency
amplitude = 0.02  # Keep volume low to avoid sub overload

def generate_elf_wave():
    t = np.arange(chunk_size) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)

def callback(outdata, frames, time_info, status):
    wave = generate_elf_wave()
    outdata[:] = wave.reshape(-1, 1)

try:
    print("🔊 Playing ELF tone (10 Hz). Press Ctrl+C to stop.")
    with sd.OutputStream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ ELF tone stopped.")

