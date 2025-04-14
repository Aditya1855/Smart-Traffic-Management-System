import tkinter as tk
from tkinter import Label, Button, Canvas, Scrollbar
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import threading
import datetime
import os
import random
from ultralytics import YOLO
from tensorflow.keras.models import load_model

cnn_model = load_model("D:/vscode/Python/new_traffic_model.keras")
yolo_model = YOLO("yolo11m.pt")

dataset_path = "D:/vscode/Python/Final Dataset/testing"
categories = ['Empty', 'Low', 'Medium', 'High', 'Traffic Jam']

directions = ['North', 'East', 'South', 'West']
lanes = {direction: "" for direction in directions}

class_labels = ['Empty', 'Low', 'Medium', 'High', 'Traffic Jam']
congestion_weights = {label: idx for idx, label in enumerate(class_labels)}
signal_timings = [10, 15, 25, 35, 45]

root = tk.Tk()
root.title("Intelligent Traffic Management System Using Computer Vision")
root.geometry("1024x2048")

canvas = tk.Canvas(root)
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas)
canvas_frame = canvas.create_window((0, 0), window=frame, anchor='nw')
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")
frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

lane_frames, lane_images, lane_labels, lane_signals, lane_counts, lane_buttons, signal_canvases = {}, {}, {}, {}, {}, {}, {}
emergency_override = {lane: None for lane in lanes}
manual_control = False
force_event = threading.Event()
manual_label = None
timer_labels = {}
served_lanes = set()

control_frame = tk.Frame(root)
control_frame.pack(side="right", fill="y", padx=10, pady=20)

def emergency(lane, signal):
    global manual_control
    emergency_override[lane] = signal
    force_event.set()
    if signal == 'red':
        manual_control = True
        manual_label.config(text="Mode: MANUAL", fg="red")
        for l in lanes:
            update_signal(l, 'red')
    elif signal == 'green':
        manual_label.config(text="Mode: MANUAL", fg="red")

def resume_auto():
    global manual_control, served_lanes
    manual_control = False
    served_lanes.clear()
    manual_label.config(text="Mode: AUTO", fg="green")

def update_signal(lane, color):
    canvas = signal_canvases[lane]
    canvas.delete("all")
    colors = {'red': 'gray', 'yellow': 'gray', 'green': 'gray'}
    if color == 'red': colors['red'] = 'red'
    elif color == 'yellow': colors['yellow'] = 'yellow'
    elif color == 'green': colors['green'] = 'green'

    canvas.create_oval(10, 10, 50, 50, fill=colors['red'], outline="black")
    canvas.create_oval(10, 60, 50, 100, fill=colors['yellow'], outline="black")
    canvas.create_oval(10, 110, 50, 150, fill=colors['green'], outline="black")

def build_gui():
    global manual_label
    for idx, lane in enumerate(lanes):
        subframe = tk.Frame(frame, bd=2, relief=tk.RIDGE, padx=5, pady=5)
        subframe.grid(row=idx//2, column=idx%2, padx=10, pady=10, sticky="n")

        container = tk.Frame(subframe)
        container.pack()

        signal_canvas = Canvas(container, width=60, height=170, bg='black')
        signal_canvas.pack(side=tk.LEFT, padx=5)
        signal_canvases[lane] = signal_canvas
        update_signal(lane, 'red')

        img_label = Label(container)
        img_label.pack(side=tk.LEFT)

        info_label = Label(subframe, text=f"{lane}: Processing...", font=("Arial", 12))
        info_label.pack()

        count_label = Label(subframe, text="Vehicles: 0", font=("Arial", 10))
        count_label.pack()

        timer_label = Label(subframe, text="", font=("Arial", 10, "bold"))
        timer_label.pack()
        timer_labels[lane] = timer_label

        button_frame = tk.Frame(subframe)
        button_frame.pack(pady=5)
        Button(button_frame, text=f"Force GREEN", bg="lightgreen", command=lambda l=lane: emergency(l, 'green')).pack(side=tk.LEFT, padx=2)
        Button(button_frame, text=f"Force RED", bg="tomato", command=lambda l=lane: emergency(l, 'red')).pack(side=tk.LEFT, padx=2)

        lane_frames[lane] = subframe
        lane_images[lane] = img_label
        lane_labels[lane] = info_label
        lane_counts[lane] = count_label
        lane_signals[lane] = signal_canvas

    Label(control_frame, text="Controls", font=("Arial", 14, "bold")).pack(pady=10)
    Button(control_frame, text="Resume AUTO", bg="lightblue", font=("Arial", 12), command=resume_auto).pack(pady=5)
    manual_label = Label(control_frame, text="Mode: AUTO", font=("Arial", 12, "bold"), fg="green")
    manual_label.pack(pady=5)

build_gui()

log_file = open("traffic_log.txt", "a")

def log(text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {text}\n")
    log_file.flush()

def preprocess_image(img):
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img_resized, axis=0)

def analyze_lane(path):
    img = cv2.imread(path)
    if img is None:
        return "Empty", 0, np.zeros((240, 320, 3), dtype=np.uint8)

    cnn_input = preprocess_image(img)
    prediction = cnn_model.predict(cnn_input)[0]
    congestion_idx = np.argmax(prediction)
    congestion_level = class_labels[congestion_idx]

    results = yolo_model.predict(source=img, imgsz=640, conf=0.25, verbose=False)[0]
    boxes = results.boxes

    count = 0
    for box in boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    display_img = cv2.resize(img, (320, 240))
    return congestion_level, count, display_img

def run_cycle(lane, level):
    global served_lanes

    if manual_control:
        if emergency_override[lane] == 'green':
            update_signal(lane, 'green')
            log(f"[MANUAL] Force GREEN: {lane}")
            timer_labels[lane]['text'] = "MANUAL GREEN"
            while manual_control and emergency_override[lane] == 'green':
                root.update()
                time.sleep(0.1)

        update_signal(lane, 'yellow')
        for t in range(3, 0, -1):
            timer_labels[lane]['text'] = f"YELLOW (to red): {t}s"
            root.update()
            time.sleep(1)
        update_signal(lane, 'red')
        timer_labels[lane]['text'] = ""
        return

    green_time = signal_timings[class_labels.index(level)]

    for l in lanes:
        update_signal(l, 'red')
        timer_labels[l]['text'] = ""

    update_signal(lane, 'yellow')
    for t in range(2, 0, -1):
        timer_labels[lane]['text'] = f"YELLOW (to green): {t}s"
        root.update()
        time.sleep(1)

    update_signal(lane, 'green')
    log(f"Green: {lane} ({level}) for {green_time}s")
    for t in range(green_time, 0, -1):
        if force_event.is_set():
            force_event.clear()
            return
        timer_labels[lane]['text'] = f"GREEN: {t}s"
        root.update()
        time.sleep(1)

    update_signal(lane, 'yellow')
    for t in range(2, 0, -1):
        timer_labels[lane]['text'] = f"YELLOW (to red): {t}s"
        root.update()
        time.sleep(1)

    update_signal(lane, 'red')
    timer_labels[lane]['text'] = ""
    served_lanes.add(lane)

def update_lane_images():
    for direction in lanes:
        category = random.choice(categories)
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            continue
        image_file = random.choice(files)
        lanes[direction] = os.path.join(folder, image_file)

def update_gui():
    traffic_data = {}

    for lane, path in lanes.items():
        level, count, display_img = analyze_lane(path)
        score = (congestion_weights[level] * 3) + (count * 5)
        traffic_data[lane] = {"level": level, "count": count, "score": score}

        lane_labels[lane]['text'] = f"{lane}: {level}"
        lane_counts[lane]['text'] = f"Vehicles: {count}"
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        lane_images[lane].configure(image=img_pil)
        lane_images[lane].image = img_pil

    for lane in lanes:
        if emergency_override[lane] == 'green':
            run_cycle(lane, traffic_data[lane]['level'])
            emergency_override[lane] = None
            return
        elif emergency_override[lane] == 'red':
            log(f"[EMERGENCY] Force RED: {lane} - Skipped")
            emergency_override[lane] = None

    if manual_control:
        return

    for _ in range(len(lanes)):
        unserved = [l for l in lanes if l not in served_lanes]
        if not unserved:
            served_lanes.clear()
            unserved = list(lanes.keys())

        sorted_lanes = sorted(unserved, key=lambda l: traffic_data[l]['score'], reverse=True)
        if sorted_lanes:
            run_cycle(sorted_lanes[0], traffic_data[sorted_lanes[0]]['level'])
            break

def loop():
    while True:
        update_lane_images()
        update_gui()
        time.sleep(2)

threading.Thread(target=loop, daemon=True).start()
root.mainloop()
log_file.close()