# Citrus Disease Detection API and AI Assistant

An intelligent backend service built with Python and Flask, combining Computer Vision with Large Language Models (LLMs) to diagnose citrus plant diseases and deliver contextual agricultural advice.

## Key Features

* **Image Classification (`/predict`):** Processes uploaded images of citrus leaves and fruits through a custom-trained TensorFlow/Keras MobileNetV2 model to detect nine distinct health states (including Canker, Greening, Black Spot, and Healthy).
* **Smart Agricultural Chatbot (`/api/chat`):** Integrates with Google's Gemini 2.5 Flash model. The chatbot is context-aware; if a leaf is diagnosed with a specific disease, the assistant utilizes that diagnosis as context to answer follow-up questions (e.g., "How do I treat this condition?").
* **Confidence Thresholding:** Automatically flags uncertain predictions (confidence < 45%) to ensure users receive highly reliable information.

## Technology Stack

* **Framework:** Python, Flask, Flask-CORS
* **Machine Learning:** TensorFlow / Keras (MobileNetV2 architecture)
* **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash)
* **Image Processing:** Pillow (PIL), NumPy

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bilalneedscodes/Citrus-Disease-Detection.git](https://github.com/bilalneedscodes/Citrus-Disease-Detection.git)
   cd Citrus-Disease-Detection/backend
