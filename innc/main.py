import sounddevice as sd
import numpy as np

fs = 44100  # Sample rate
chunk_duration = 0.1  # seconds, shorter duration reduces latency
gain = 1.0  # Volume gain for cancellation (adjust as needed)

print("🎙️ Real-time Active Noise Cancellation started. Press Ctrl+C to stop.")

try:
    while True:
        # Record audio chunk from mic
        mic_audio = sd.rec(int(fs * chunk_duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        # Invert audio waveform (phase inversion)
        inverted_audio = -gain * mic_audio

        # Play back inverted audio
        sd.play(inverted_audio, samplerate=fs)
        sd.wait()

except KeyboardInterrupt:
    print("\n⏹️ ANC stopped.")

