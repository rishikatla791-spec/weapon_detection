# 🛡️ WeaponShield AI: Enhanced Weapon Detection System

WeaponShield AI is a hybrid weapon detection system that combines the real-time speed of **YOLO (You Only Look Once)** with the advanced visual reasoning of **Google Gemini 1.5 Flash**. This multi-layered approach significantly reduces false positives by using a "second opinion" for verification.

## 🚀 Key Features

- **Hybrid Detection Engine:** Uses YOLOv3 for initial fast scanning and Gemini 1.5 Flash for high-confidence verification.
- **Visual Classification:** Beyond simple detection, it attempts to classify the specific type (Pistol, Rifle, Knife, etc.) using geometric heuristics and AI.
- **Multi-Source Support:** Process static images or video files (MP4, AVI, etc.) with real-time frame analysis.
- **Modern Dashboard:** A clean, responsive web interface built with Flask and Vanilla CSS/JS.
- **Confidence Scoring:** Provides detection confidence levels for both YOLO and Gemini verification.

## 🛠️ Technology Stack

- **Backend:** Python (Flask, OpenCV)
- **Deep Learning:** YOLOv3 (weights included via Git LFS)
- **Generative AI:** Google Gemini 1.5 Flash API
- **Frontend:** HTML5, CSS3, JavaScript
- **Environment:** Python 3.x

## 📋 Prerequisites

- **Python 3.8+**
- **Google AI API Key:** To enable Gemini verification (optional but recommended).
- **Git LFS:** Required to download the large YOLO weight files after cloning.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rishikatla791-spec/weapon_detection.git
   cd weapon_detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_api_key_here
   ```

4. **Model Weights:**
   Ensure `yolov3_training_2000.weights` is in the root directory. (If using Git LFS, run `git lfs pull`).

## 🖥️ Usage

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
2. **Access the application:**
   Open `http://localhost:5004` in your web browser.
3. **Analyze:**
   Upload an image or video to see the detection in action.

## 📂 Project Structure

```text
├── app.py                # Main Flask application
├── weapon_detection.py    # Core detection logic
├── yolov3_testing.cfg     # YOLO architecture configuration
├── weapon.names           # Class labels (weapon)
├── static/                # CSS, JS, and UI assets
├── templates/             # HTML templates
├── weights/               # Alternative weights storage
└── uploads/               # Temporary storage for processed files
```

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Disclaimer: This project is for educational and research purposes. It should not be used as the sole source of security monitoring.*
