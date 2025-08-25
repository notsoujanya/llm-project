# app.py
# To run this:
# 1. Install libraries: pip install Flask flask-cors transformers torch sentencepiece requests
# 2. Run from terminal: python app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
# It's best to set your Gemini API key as an environment variable.
# Example in terminal: export GEMINI_API_KEY='your_api_key_here'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

# --- Initialize Flask App ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow the frontend to talk to this backend
CORS(app)

# --- Load NLP Models and Tokenizers from Scratch ---
# This approach gives us more control than the pipeline.
try:
    print("Loading models and tokenizers... This may take a few moments.")
    
    # Load BART Model
    bart_model_name = "sshleifer/distilbart-cnn-12-6"
    bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name)
    
    # Load T5 Model
    t5_model_name = "t5-small"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

    print("Models and tokenizers loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}. Please ensure you have an internet connection and all libraries are installed.")
    bart_tokenizer, bart_model, t5_tokenizer, t5_model = None, None, None, None


# --- API Endpoints ---

@app.route('/api/gemini', methods=['POST'])
def handle_gemini():
    """Proxies the request to the Gemini API to keep the API key secure."""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server"}), 500

    payload = {"contents": [{"parts": [{"text": data['prompt']}]}]}
    
    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()
        gemini_data = response.json()
        text = gemini_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        return jsonify({"text": text})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to communicate with Gemini API: {e}"}), 502
    except (KeyError, IndexError):
        return jsonify({"error": "Unexpected response format from Gemini API"}), 500

@app.route('/api/summarize-bart', methods=['POST'])
def handle_bart_summarize():
    """Generates a summary using the DistilBART model from scratch."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    if not bart_model:
        return jsonify({"error": "BART model is not available"}), 500
        
    text = data['text']
    min_len = max(10, len(text.split()) // 8)
    max_len = max(20, len(text.split()) // 4)

    try:
        # 1. Tokenize: Convert the text into input IDs the model understands.
        inputs = bart_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

        # 2. Generate: Pass the tokenized inputs to the model to get output IDs.
        summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=max_len, min_length=min_len, early_stopping=True)

        # 3. Decode: Convert the output IDs back into a human-readable string.
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return jsonify({"text": summary})
    except Exception as e:
        return jsonify({"error": f"Failed to generate summary with BART: {e}"}), 500

@app.route('/api/summarize-t5', methods=['POST'])
def handle_t5_summarize():
    """Generates a summary using the T5-small model from scratch."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    if not t5_model:
        return jsonify({"error": "T5 model is not available"}), 500

    text = data['text']
    # T5 requires a prefix for the task
    input_text = "summarize: " + text
    min_len = max(10, len(text.split()) // 8)
    max_len = max(20, len(text.split()) // 4)
    
    try:
        # 1. Tokenize: Convert the text into input IDs.
        inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        # 2. Generate: Pass the tokenized inputs to the model.
        summary_ids = t5_model.generate(inputs.input_ids, num_beams=4, max_length=max_len, min_length=min_len, early_stopping=True)
        
        # 3. Decode: Convert the output IDs back into a string.
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"text": summary})
    except Exception as e:
        return jsonify({"error": f"Failed to generate summary with T5: {e}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)