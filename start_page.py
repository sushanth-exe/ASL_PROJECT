import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import multiprocess
import subprocess
import threading
import time
import os
import shutil

def start_application():
    subprocess.run("python ./multiprocess.py")
    # time.sleep(17)
    f = open("./temp/text.txt", "r")
    for x in f:
        title_label = tk.Label(window, text=x, font=("Consolas", 15), borderwidth=0, background="#5c6b6c")
        title_label.pack()

def animate_button(event):
    start_button.configure(relief="sunken")

def animate_button_release(event):
    start_button.configure(relief="raised")

# Create the main window
window = tk.Tk()
window.title("SIGN SENSEI")

# Set the window size
with open("text.txt", "w") as file:
    file.write('')
window_width = 1280
window_height = 720
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

window.config(bg = "#5c6b6c")

# Load and display the image
image_path = ".\logo.jpg"  # Replace with the path to your image
image = Image.open(image_path)
image = image.resize((900, 450))
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(window, image=photo, borderwidth=0, padx = 0, pady = 0)
image_label.pack(pady=5)

# Create the title label
title_label = tk.Label(window, text="Welcome to SIGN SENSEI", font=("Consolas", 20), borderwidth=0, padx = 0, pady = 0, background="#5c6b6c")
title_label.pack()

# Create the description label
description_label = tk.Label(window, text="ASL Language Detection", font=("Consolas", 14), wraplength=400, justify="center", background="#5c6b6c")
description_label.pack(pady=10)

# Create the start button
start_button = tk.Button(window, text="START", command=start_application, font=("Consolas", 12))
start_button.pack()
# Add animation to the start button
start_button.bind("<Enter>", animate_button)
start_button.bind("<Leave>", animate_button_release)
os.mkdir("temp")
window.mainloop()
shutil.rmtree("temp")

# Start the GUI event loop

