import os
import json
import torch
from flask import Flask, request, jsonify, render_template
from model.model import SentimentLSTM
from model.vocabulary import Vocabulary, tokenize
from model.dataset import pad_sequence  # reuse pad helper

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

CHECKPOINT_DIR = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    cfg_path   = os.path.join(CHECKPOINT_DIR, "config.json")
    vocab_path = os.path.join(CHECKPOINT_DIR, "vocab.json")
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    with open(cfg_path) as f:
        cfg = json.load(f)

    vocab = Vocabulary.load(vocab_path)

    model = SentimentLSTM(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vocab, cfg["max_len"]


try:
    MODEL, VOCAB, MAX_LEN = load_model()
    MODEL_READY = True
    print("Model loaded successfully.")
except FileNotFoundError:
    MODEL_READY = False
    print("WARNING: No checkpoint found. Run train.py first.")


def predict(text: str):
    tokens  = tokenize(text)
    indices = VOCAB.encode(tokens)
    padded  = pad_sequence(indices, MAX_LEN)

    tensor  = torch.tensor([padded], dtype=torch.long).to(device)

    with torch.no_grad():
        prob, attn_weights = MODEL(tensor)

    score = prob.item()
    label = "Positive 😊" if score >= 0.5 else "Negative 😞"
    confidence = score if score >= 0.5 else 1 - score

    # Top-5 attention tokens for explanation
    weights = attn_weights[0].cpu().tolist()
    token_weights = list(zip(tokens[:MAX_LEN], weights[:len(tokens)]))
    top_tokens = sorted(token_weights, key=lambda x: x[1], reverse=True)[:5]

    return {
        "label":      label,
        "score":      round(score, 4),
        "confidence": round(confidence * 100, 1),
        "top_tokens": [{"word": w, "weight": round(v, 4)} for w, v in top_tokens],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)."}), 400

    result = predict(text)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_READY, "device": str(device)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
