import numpy as np
import sounddevice as sd
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
chirp_visual_boost_duration = 1.0
last_chirp_visual = 0
last_baseline_chirp = 0
baseline_interval = 3.0
baseline_enabled = True
baseline_length = chunk_size
constant_enabled = True
last_constant_chirp = 0

# Configurable trail length (number of frames of past data to keep)
max_trail = 10
waveform_trail = [np.zeros(chunk_size, dtype=np.float32) for _ in range(max_trail)]
waveform_data = np.zeros((2, chunk_size), dtype=np.float32)  # [0]=input, [1]=chirp

# UI code with waveform

def launch_ui():
    def update_debounce(val):
        global chirp_debounce
        chirp_debounce = float(val)

    def update_trail(val):
        global max_trail, waveform_trail
        max_trail = int(val)
        waveform_trail = waveform_trail[-max_trail:]

    def update_baseline_length(val):
        global baseline_length
        baseline_length = int(val)
        if baseline_length > chunk_size:
            baseline_length = chunk_size

    def toggle_baseline():
        global baseline_enabled
        baseline_enabled = not baseline_enabled
        toggle_button.config(text=f"Baseline: {'On' if baseline_enabled else 'Off'}")

    def toggle_constant():
        global constant_enabled
        constant_enabled = not constant_enabled
        toggle_constant_button.config(text=f"Constant: {'On' if constant_enabled else 'Off'}")

    def update_plot():
        while True:
            ax.clear()
            ax.set_ylim(-0.8, 0.8)
            ax.set_title("Input and Chirp Waveform Viewer with Trails")
            for i, trail_wave in enumerate(waveform_trail[-max_trail:]):
                alpha = min(1.0, 0.1 * (i + 1))
                ax.plot(trail_wave, color='blue', alpha=alpha)
            ax.plot(waveform_data[0], color='blue', label="Input")
            chirp_mag = 4.0 if (time.time() - last_chirp_visual) < chirp_visual_boost_duration else 1.0
            ax.plot(waveform_data[1] * chirp_mag, color='red', linewidth=3 if chirp_mag > 1 else 1, label="Chirp")
            ax.legend()
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

    tk.Label(root, text="Waveform Trail Length").pack()
    trail_slider = tk.Scale(root, from_=1, to=10, resolution=1,
                            orient=tk.HORIZONTAL, command=update_trail)
    trail_slider.set(max_trail)
    trail_slider.pack()

    toggle_button = tk.Button(root, text="Baseline: On", command=toggle_baseline)
    toggle_button.pack()

    tk.Label(root, text="Baseline Chirp Length").pack()
    baseline_length_slider = tk.Scale(root, from_=256, to=1024, resolution=128,
                                      orient=tk.HORIZONTAL, command=update_baseline_length)
    baseline_length_slider.set(baseline_length)
    baseline_length_slider.pack()

    toggle_constant_button = tk.Button(root, text="Constant: On", command=toggle_constant)
    toggle_constant_button.pack()

    fig, ax = plt.subplots(figsize=(7, 3))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    plot_thread = Thread(target=update_plot, daemon=True)
    plot_thread.start()

    root.mainloop()

ui_thread = Thread(target=launch_ui, daemon=True)
ui_thread.start()

def generate_derivative_chirp(signal):
    derivative = np.diff(signal, prepend=signal[0])
    chirp_signal = 0.1 * derivative / np.max(np.abs(derivative) + 1e-6)
    return chirp_signal.astype(np.float32)

def generate_baseline_chirp(length):
    length = min(length, chunk_size)
    t = np.arange(length) / fs
    chirp = 0.05 * np.sin(2 * np.pi * 5000 * t)
    return np.pad(chirp, (0, chunk_size - len(chirp)), 'constant').astype(np.float32)

def generate_constant_chirp():
    t = np.arange(chunk_size) / fs
    f1, f2 = 300, 8000
    chirp = 0.05 * np.sin(2 * np.pi * f1 * t) + 0.05 * np.sin(2 * np.pi * f2 * t)
    return chirp.astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global prev_magnitudes, last_chirp_time, last_chirp_visual, waveform_data, waveform_trail
    global last_baseline_chirp, last_constant_chirp

    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)
    hash_digest = hashlib.sha256(input_audio.tobytes()).hexdigest()
    hash_matched = hash_digest in hash_history
    if not hash_matched:
        hash_history.append(hash_digest)
        if len(hash_history) > 20:
            hash_history.pop(0)

    chirp_signal = np.zeros(frames, dtype=np.float32)
    now = time.time()

    if constant_enabled and (now - last_constant_chirp) > chirp_debounce:
        chirp_signal = generate_constant_chirp()
        last_constant_chirp = now
        last_chirp_visual = now

    elif baseline_enabled and (now - last_baseline_chirp) > baseline_interval:
        chirp_signal = generate_baseline_chirp(baseline_length)
        last_baseline_chirp = now
        last_chirp_visual = now

    elif hash_matched and (now - last_chirp_time) > chirp_debounce:
        chirp_signal = generate_derivative_chirp(input_audio)
        last_chirp_time = now
        last_chirp_visual = now

    waveform_data[0] = input_audio[:frames]
    waveform_data[1] = chirp_signal[:frames]

    waveform_trail.append(input_audio[:frames])
    if len(waveform_trail) > max_trail:
        waveform_trail = waveform_trail[-max_trail:]

    outdata[:, 0] = chirp_signal[:frames]

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=chunk_size, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Gitarg Hash Chirper Stopped.")

