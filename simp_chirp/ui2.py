import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from scipy.fft import rfft, rfftfreq
import hashlib
import time
import tkinter as tk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fs = 44100
chunk_size = 1024
print("📡 Gitarg Hash Chirper with UI and Waveform Viewer Activated. Ctrl+C to stop.")

hash_history = []
prev_magnitudes = None
last_chirp_time = 0
chirp_debounce = 3

# Waveform data shared with UI
waveform_data = np.zeros((2, chunk_size), dtype=np.float32)  # [0]=input, [1]=chirp

# UI code with waveform

def launch_ui():
    def update_debounce(val):
        global chirp_debounce
        chirp_debounce = float(val)

    def update_plot():
        while True:
            line_input.set_ydata(waveform_data[0])
            line_chirp.set_ydata(waveform_data[1])
            canvas.draw()
            canvas.flush_events()
            time.sleep(0.05)

    root = tk.Tk()
    root.title("Gitarg Chirp Control")

    tk.Label(root, text="Chirp Debounce (seconds)").pack()
    debounce_slider = tk.Scale(root, from_=0.1, to=10.0, resolution=0.1,
                               orient=tk.HORIZONTAL, command=update_debounce)
    debounce_slider.set(chirp_debounce)
    debounce_slider.pack()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("Input and Chirp Waveform Viewer")
    ax.set_ylim(-0.2, 0.2)
    line_input, = ax.plot(waveform_data[0], label="Input")
    line_chirp, = ax.plot(waveform_data[1], label="Chirp")
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    plot_thread = Thread(target=update_plot, daemon=True)
    plot_thread.start()

    root.mainloop()

ui_thread = Thread(target=launch_ui, daemon=True)
ui_thread.start()

def generate_chirp(f0=3000, f1=1000):
    chirp_t = np.arange(chunk_size) / fs
    chirp_tone = 0.1 * chirp(chirp_t, f0=f0, f1=f1, t1=chirp_t[-1], method='linear')
    return chirp_tone.astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global prev_magnitudes, last_chirp_time, waveform_data

    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)
    hash_digest = hashlib.sha256(input_audio.tobytes()).hexdigest()
    hash_matched = hash_digest in hash_history
    if not hash_matched:
        hash_history.append(hash_digest)
        if len(hash_history) > 20:
            hash_history.pop(0)

    chirp_signal = np.zeros(frames, dtype=np.float32)
    now = time.time()

    if prev_magnitudes is not None and hash_matched and (now - last_chirp_time) > chirp_debounce:
        Y = rfft(input_audio)
        freqs = rfftfreq(len(input_audio), 1/fs)
        magnitudes = np.abs(Y)
        sorted_indices = np.argsort(magnitudes)[::-1]
        if len(sorted_indices) > 1:
            f0 = freqs[sorted_indices[0]]
            f1 = freqs[sorted_indices[1]]
            chirp_signal = generate_chirp(f0=f0, f1=f1)
            last_chirp_time = now
        prev_magnitudes = magnitudes
    else:
        Y = rfft(input_audio)
        prev_magnitudes = np.abs(Y)

    waveform_data[0] = input_audio[:frames]
    waveform_data[1] = chirp_signal[:frames]
    outdata[:, 0] = chirp_signal[:frames]

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Hash Chirper Stopped.")
