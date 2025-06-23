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
print("📱 Gitarg Hash Chirper with UI and Waveform Viewer Activated. Ctrl+C to stop.")

hash_history = []
hash_timestamps = {}
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
constant_chirp_speed = 1.0

max_trail = 10
waveform_trail = [np.zeros(chunk_size, dtype=np.float32) for _ in range(max_trail)]
waveform_data = np.zeros((2, chunk_size), dtype=np.float32)

ui_elements = {}

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

    def update_constant_speed(val):
        global constant_chirp_speed
        constant_chirp_speed = float(val)

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

    tk.Label(root, text="Constant Chirp Speed Multiplier").pack()
    speed_slider = tk.Scale(root, from_=0.5, to=5.0, resolution=0.1,
                            orient=tk.HORIZONTAL, command=update_constant_speed)
    speed_slider.set(constant_chirp_speed)
    speed_slider.pack()

    fig, ax = plt.subplots(figsize=(7, 3))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    plot_thread = Thread(target=update_plot, daemon=True)
    plot_thread.start()

    root.mainloop()

ui_thread = Thread(target=launch_ui, daemon=True)
ui_thread.start()

def generate_segmented_chirp(input_wave):
    t = np.arange(chunk_size) / fs
    chirp = np.zeros_like(input_wave)
    segment_size = chunk_size // 10
    for i in range(10):
        start = i * segment_size
        end = start + segment_size
        segment = input_wave[start:end]
        if len(segment) == 0:
            continue
        min_val = np.min(segment)
        max_val = np.max(segment)
        f = 300 + constant_chirp_speed * abs(max_val - min_val) * 5000
        chirp[start:end] = 0.05 * np.sin(2 * np.pi * f * t[start:end])
    return chirp.astype(np.float32)

def callback(indata, outdata, frames, time_info, status):
    global waveform_data, waveform_trail

    input_audio = indata[:, 0] if indata is not None else np.zeros(frames)
    chirp_signal = generate_segmented_chirp(input_audio)

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

