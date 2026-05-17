import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io, os

from google import genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

print("Loading image model... please wait.")
try:
    MODEL_PATH = 'models/lemon_leaf_disease_model_accurate.h5'
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# 2. CONFIGURE NEW GEMINI CLIENT

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("No GEMINI_API_KEY found. Please check your .env file.")

gemini_client = genai.Client(api_key=api_key)


SYSTEM_INSTRUCTION = """
You are an expert agricultural assistant specializing in citrus plant health. 
Answer questions related to citrus diseases like Anthracnose, Black Spot, Canker, Greening, etc. 
Keep answers concise, helpful, and easy to understand for farmers.
"""


# 3. LABELS & CONFIGURATION

LABELS = [
    'Anthracnose', 
    'Black Spot', 
    'Citrus Canker', 
    'Citrus Hindu Mite', 
    'Curl Leaf', 
    'Dry Leaf', 
    'Greening', 
    'Healthy', 
    'Melanose'
]

CONFIDENCE_THRESHOLD = 0.45


# 4. PREPROCESSING

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224)) 
    img_array = np.array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


# 5. PREDICTION ENDPOINT

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    
    try:
        input_data = prepare_image(img_bytes)
        pred = model.predict(input_data)
        
        label_idx = np.argmax(pred)
        confidence = float(pred[0][label_idx])
        predicted_label = LABELS[label_idx]
        
        diagnosis_result = predicted_label
        advice = ""
        
        if confidence < CONFIDENCE_THRESHOLD:
            diagnosis_result = "Uncertain / Unclear"
            advice = "The model is not sure (Confidence < 45%). Please try uploading a clearer image."
        elif predicted_label == "Healthy":
            advice = "✅ Your plant looks healthy! Keep up the good work."
        else:
            if confidence < 0.60:
                advice = f"⚠️ Potentially {predicted_label}, but the confidence is medium ({confidence*100:.1f}%). Check closely."
            else:
                advice = f"⚠️ Detected {predicted_label}. You should isolate this plant and check for treatment options."

        print(f"Prediction: {predicted_label} ({confidence*100:.2f}%)")

        return jsonify({
            "detected_type": "Leaf",
            "diagnosis": diagnosis_result,
            "confidence": f"{confidence * 100:.2f}%",
            "advice": advice
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# 6. CHATBOT ENDPOINT (UPDATED SYNTAX)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    current_diagnosis = data.get('diagnosis') 
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        context_str = ""
        if current_diagnosis:
            context_str = f"\n[System Context: The user just scanned a leaf and the system diagnosed it as '{current_diagnosis}'. Use this context if they ask questions like 'how do I treat this?' or 'what is this?']\n"

        full_prompt = f"{SYSTEM_INSTRUCTION}{context_str}\nUser asks: {user_message}"
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        
        return jsonify({"reply": response.text})
    
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({"error": "Could not connect to the AI service. Please check your API key and terminal logs."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)