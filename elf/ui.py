import numpy as np
import sounddevice as sd
import tkinter as tk
from threading import Thread

# Audio configuration
fs = 44100
chunk_size = 1024
current_freq = 10.0
current_volume = 0.05

def generate_wave(freq, volume):
    t = np.arange(chunk_size) / fs
    return (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def audio_callback(outdata, frames, time_info, status):
    wave = generate_wave(current_freq, current_volume)
    outdata[:] = wave.reshape(-1, 1)

def launch_ui():
    def update_freq_display():
        freq_label.config(text=f"Freq: {current_freq:.2f} Hz")
        freq_slider.set(current_freq)
        freq_entry_var.set(f"{current_freq:.2f}")

    def on_freq_slider(val):
        global current_freq
        current_freq = float(val)
        update_freq_display()

    def on_freq_entry(event=None):
        global current_freq
        try:
            val = float(freq_entry_var.get())
            if 0 <= val <= 30000:
                current_freq = val
                update_freq_display()
        except ValueError:
            pass  # Invalid input; ignore

    def increment_freq():
        global current_freq
        if current_freq < 30000:
            current_freq += 1
            update_freq_display()

    def decrement_freq():
        global current_freq
        if current_freq > 0:
            current_freq -= 1
            update_freq_display()

    def on_vol_change(val):
        global current_volume
        current_volume = float(val)
        vol_label.config(text=f"Volume: {current_volume:.2f}")

    root = tk.Tk()
    root.title("🎛️ Tone Generator with Direct Input")

    # Frequency control
    tk.Label(root, text="Frequency (0–30000 Hz)").pack()

    freq_frame = tk.Frame(root)
    freq_frame.pack()

    dec_button = tk.Button(freq_frame, text="◀", width=3, command=decrement_freq)
    dec_button.pack(side=tk.LEFT)

    freq_entry_var = tk.StringVar(value=f"{current_freq:.2f}")
    freq_entry = tk.Entry(freq_frame, textvariable=freq_entry_var, width=10, justify='center')
    freq_entry.bind("<Return>", on_freq_entry)
    freq_entry.pack(side=tk.LEFT, padx=5)

    inc_button = tk.Button(freq_frame, text="▶", width=3, command=increment_freq)
    inc_button.pack(side=tk.LEFT)

    freq_label = tk.Label(root, text=f"Freq: {current_freq:.2f} Hz")
    freq_label.pack()

    freq_slider = tk.Scale(root, from_=0, to=30000, resolution=1, orient=tk.HORIZONTAL, length=500, command=on_freq_slider)
    freq_slider.set(current_freq)
    freq_slider.pack()

    # Volume control
    tk.Label(root, text="Volume (0.0 – 1.0)").pack()
    vol_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=500, command=on_vol_change)
    vol_slider.set(current_volume)
    vol_slider.pack()
    vol_label = tk.Label(root, text=f"Volume: {current_volume:.2f}")
    vol_label.pack()

    root.mainloop()

# Launch the UI thread
Thread(target=launch_ui, daemon=True).start()

# Audio stream
try:
    print("🔊 Use the UI to control tone. Ctrl+C to stop.")
    with sd.OutputStream(channels=1, samplerate=fs, blocksize=chunk_size, callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n⏹️ Stopped.")

