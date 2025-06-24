import numpy as np
import sounddevice as sd
import threading
import time

fs = 44100
chunk_size = 1024

# Shared parameters
params = {
    'frequency': 440.0,
    'volume': 0.1,
    'running': True
}

def generate_wave(freq, volume, length):
    t = np.arange(length) / fs
    return (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def audio_loop():
    def callback(outdata, frames, time_info, status):
        wave = generate_wave(params['frequency'], params['volume'], frames)
        outdata[:] = wave.reshape(-1, 1)

    with sd.OutputStream(samplerate=fs, channels=1, blocksize=chunk_size, callback=callback):
        while params['running']:
            time.sleep(0.1)

# Start audio in background thread
threading.Thread(target=audio_loop, daemon=True).start()

# Interactive CLI loop
try:
    print("🔊 Real-time Tone Generator")
    print("Enter new frequency and volume anytime (or 'q' to quit).")
    while True:
        freq_input = input("\nFrequency (Hz): ")
        if freq_input.lower() == 'q':
            break
        vol_input = input("Volume (0.0–1.0): ")
        if vol_input.lower() == 'q':
            break
        try:
            new_freq = float(freq_input)
            new_vol = float(vol_input)
            if 0 <= new_freq <= 30000 and 0.0 <= new_vol <= 1.0:
                params['frequency'] = new_freq
                params['volume'] = new_vol
                print(f"🎚️ Updated -> Frequency: {new_freq:.2f} Hz | Volume: {new_vol:.2f}")
            else:
                print("⚠️ Frequency must be 0–30000 Hz and volume 0.0–1.0")
        except ValueError:
            print("❌ Invalid input. Please enter numbers.")

except KeyboardInterrupt:
    pass
finally:
    params['running'] = False
    print("\n⏹️ Tone generator stopped.")

