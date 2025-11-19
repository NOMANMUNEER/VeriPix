# VeriPix: Intelligent Image Matching Server 🖼️

**VeriPix** is a local server application built with Python and Streamlit that verifies if two images are essentially the same, even if one has been cropped, resized, rotated, or compressed.

It uses a two-stage verification process:
1.  **Fast Scan:** Perceptual Hashing (aHash, dHash, wHash).
2.  **Deep Scan:** ORB Feature Detection with RANSAC Geometric Verification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green)

## 🚀 Features

* **Dual-Engine Verification:**
    * *Stage 1:* Checks perceptual hashes to find quick matches (resistant to resizing/color changes).
    * *Stage 2:* If hashes fail, it uses ORB (Oriented FAST and Rotated BRIEF) to find key features and filters outliers using RANSAC (Random Sample Consensus).
* **Interactive UI:** Adjust thresholds (Hash Distance & Minimum Inliers) in real-time using sliders.
* **Visual Feedback:** See the "Original" and "Target" images side-by-side with immediate pass/fail metrics.
* **Privacy Focused:** Images are processed locally and temporary files are cleaned up automatically.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/VeriPix.git](https://github.com/yourusername/VeriPix.git)
    cd VeriPix
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the local server with one command:

```bash
streamlit run app.py
