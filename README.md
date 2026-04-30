# 🍊 Citrus Disease Detection API & AI Assistant

An intelligent backend service built with Python and Flask that combines Computer Vision with Large Language Models (LLMs) to diagnose citrus plant diseases and provide contextual agricultural advice.

## 🚀 Features

* **Image Classification (`/predict`):** Processes uploaded images of citrus leaves/fruits through a custom-trained TensorFlow/Keras MobileNetV2 model to detect 9 different health states (including Canker, Greening, Black Spot, and Healthy).
* **Smart Agricultural Chatbot (`/api/chat`):** Integrates with Google's Gemini 2.5 Flash model. The chatbot is context-aware; if a leaf is diagnosed with a specific disease, the chatbot uses that diagnosis as context to answer user follow-up questions (e.g., "How do I treat this?").
* **Confidence Thresholding:** Automatically flags uncertain predictions (confidence < 45%) to ensure farmers receive reliable information.

## 🛠️ Tech Stack

* **Framework:** Python, Flask, Flask-CORS
* **Machine Learning:** TensorFlow / Keras (MobileNetV2 architecture)
* **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash)
* **Image Processing:** Pillow (PIL), NumPy

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Citrus-Disease-Detection.git](https://github.com/YourUsername/Citrus-Disease-Detection.git)
   cd Citrus-Disease-Detection/backend
