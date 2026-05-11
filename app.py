from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
#from flask_cors import CORS

app = Flask(__name__)
#CORS(app)
model = joblib.load("multi_intent_model.pkl")
mlb = joblib.load("mlb.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

def detect_language(text):
    if re.search(r'[\u0600-\u06FF]', text):
        return "AR"
    return "EN"

def predict_intents(text, threshold=0.6):
    #if not text.strip():
        #return {"unknown": 1.0}
    cleaned = clean_text(text)

    probs = model.predict_proba([cleaned])[0]
    label_probs = list(zip(mlb.classes_, probs))
    top_labels = sorted(label_probs, key=lambda x: x[1], reverse=True)

    result = {}

    #for i, label in enumerate(mlb.classes_):
    for label, prob in top_labels:
        if prob >= threshold:
            result[label] = float(prob)

    if len(result) == 0:
        result = {"unknown": 1.0}


    return result

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)

    if "message" not in data:
        return jsonify({"error": "message is required"}), 400

    text = data["message"]
    lang = detect_language(text)

    result = predict_intents(text)
    if result is None:
        lang = detect_language(text)
        msg = (
            "مش قادر افهم سؤالك ممكن توضح اكتر"
            if lang == "AR"
            else "I didn't understand the question"
        )
        return jsonify({
            "input": text,
            "output": {
                "unknown": 1.0
            },
            "message": msg,
            "language": lang
        })

    return jsonify({
        "input": text,
        "output": result,
        "language": lang
    })
        

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })



@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "API is running ",
        "endpoints": ["/predict"]
    })

if __name__ == "__main__":
    app.run()