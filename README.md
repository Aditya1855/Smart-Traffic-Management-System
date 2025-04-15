# 🚦 Smart Traffic Management System

An AI-powered Smart Traffic Management System that uses **YOLOv11**, **CNN-based traffic severity classification**, and **Reinforcement Learning (RL)** to dynamically control traffic signals based on real-time congestion levels.

---

## 🔍 Overview

This project aims to optimize traffic signal timings at intersections using:

- **YOLOv11** for vehicle detection.
- **CNN classifier** to categorize traffic congestion into five levels: *Empty, Low, Medium, High, Traffic Jam*.
- **Reinforcement Learning (RL)** with a custom **Gym environment** to make intelligent decisions about green light durations.
- A user interface with **manual override** and **emergency controls**.
- Integration of real-time traffic severity into RL agent's observation space.

---

## 🎯 Key Features

- 🚗 Vehicle detection with YOLOv11
- 🧠 CNN-based traffic congestion classification
- ♻️ Dynamic green signal allocation based on traffic severity
- 🕹 Manual and emergency override controls
- 📊 Logging and analytics for traffic decisions
- 🧪 Custom OpenAI Gym environment for RL training
- 🧠 Integration with RL agents like DQN or PPO

---

## 🛠 Tech Stack

- **Python**
- **YOLOv11** (Ultralytics)
- **TensorFlow / Keras** (MobileNetV2 CNN model)
- **OpenAI Gym** (custom environment)
- **Stable-Baselines3** (RL algorithms like DQN, PPO)
- **Tkinter** or **HTML/CSS Frontend** (depending on version)
- **Flask** (for backend, if web-based)

---


---

## 🚀 Getting Started

1. Clone the Repository
gh repo clone Aditya1855/Smart-Traffic-Management-System

2. Install dependencies
pip install -r requirements.txt

3. Run the simulation
python traffic_sim.py

📦 Dataset
The CNN model is trained on a custom dataset with 5 traffic severity levels:
Empty
Low
Medium
High
Traffic Jam
Data augmentation and class weighting were used to balance the model.

🧠 Reinforcement Learning
RL agents (e.g., DQN, PPO) are trained using a custom Gym environment simulating traffic lanes.
Observations are formed using CNN outputs for each lane.
The agent learns optimal timing to minimize overall waiting time and maximize flow.

📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

## 📦 Download Model and Database Files

You can download the models and database file from Google Drive using the following links:

- CNN MODEL - (https://drive.google.com/file/d/1RPsdZvV03pNDlK9eppd2epeJROxKA9Jk/view?usp=drive_link)
- YOLO MODEL - (https://drive.google.com/file/d/1YuAp_g9degw2CLluK72rrxdD-ADcWMHs/view?usp=drive_link)
- PPO RL AGENT - (https://drive.google.com/file/d/1JAO7q7lpbFgSEGIjjSLtS2ElPM8_uyQt/view?usp=drive_link)
- DATABASE - (https://drive.google.com/file/d/1HrVX_Vvuj3Zyqa0-xk7-y9FSDYt6LKbc/view?usp=drive_link)

🙌 Acknowledgements
YOLOv11 by Ultralytics
TensorFlow/Keras
OpenAI Gym
Stable-Baselines3

