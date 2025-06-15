import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

duration = 5  # seconds to record ambient noise
fs = 44100    # Sampling frequency

print("🎙️ Recording ambient noise...")
noise_recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("✅ Ambient noise recording completed!")

# Save noise profile for reuse (optional)
wav.write('noise_profile.wav', fs, noise_recording)

