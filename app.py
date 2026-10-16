from pathlib import Path
import io, json
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.pt"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def load_model():
    global model
    if not MODEL_PATH.exists():
        return None
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    name = ckpt.get("architecture", "mobilenet_v3_small")
    if name == "resnet18":
        net = models.resnet18(weights=None); net.fc = torch.nn.Linear(net.fc.in_features, len(CLASSES))
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None); net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(CLASSES))
    else:
        net = models.mobilenet_v3_small(weights=None); net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, len(CLASSES))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    model = net
    return model

@app.route("/")
def index():
    return render_template("index.html", ready=MODEL_PATH.exists())

@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify(error="Choose at least one image."), 400
    net = model or load_model()
    if net is None:
        return jsonify(error="No trained model found yet. Follow README.md to run train.py first."), 503
    output=[]
    for f in request.files.getlist("images"):
        if not f.filename:
            continue
        try:
            img = Image.open(f.stream).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(net(x), dim=1)[0].cpu().tolist()
            idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            output.append({"filename": f.filename, "prediction": CLASSES[idx],
                           "confidence": round(probs[idx]*100, 2),
                           "scores": {CLASSES[i]: round(probs[i]*100, 2) for i in range(len(CLASSES))}})
        except Exception:
            output.append({"filename": f.filename, "error": "Could not read this file as an image."})
    return jsonify(results=output, disclaimer="Educational prediction only; not a diagnosis.")

if __name__ == "__main__":
    if MODEL_PATH.exists(): load_model()
    app.run(debug=True)
