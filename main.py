import sounddevice as sd
import noisereduce as nr
import numpy as np

# Configuration
fs = 44100                # Sampling frequency
duration = 1              # Duration of audio chunks (seconds)

# Load or record noise profile
print("🎙️ Recording ambient noise for profile...")
noise_sample = sd.rec(int(5 * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
noise_sample = noise_sample.flatten()
print("✅ Noise profile ready!")

print("🔊 Starting real-time noise reduction. Press Ctrl+C to stop.")
try:
    while True:
        # Record audio chunk
        audio_chunk = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_chunk = audio_chunk.flatten()

        # Noise reduction
        reduced_noise_audio = nr.reduce_noise(y=audio_chunk, y_noise=noise_sample, sr=fs, prop_decrease=0.88)

        # Playback noise-reduced audio
        sd.play(reduced_noise_audio, fs)
        sd.wait()

except KeyboardInterrupt:
    print("\n⏹️ Real-time noise reduction stopped.")

