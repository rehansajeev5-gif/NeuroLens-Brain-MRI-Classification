
from pathlib import Path
import base64
import io

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.pt"

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None
target_layer = None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

def load_model():
    global model, target_layer

    if not MODEL_PATH.exists():
        return None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    architecture = checkpoint.get(
        "architecture",
        "mobilenet_v3_small"
    )

    if architecture == "resnet18":
        net = models.resnet18(weights=None)

        net.fc = torch.nn.Linear(
            net.fc.in_features,
            len(CLASSES)
        )

        target_layer = net.layer4[-1]

    elif architecture == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)

        net.classifier[1] = torch.nn.Linear(
            net.classifier[1].in_features,
            len(CLASSES)
        )

        target_layer = net.features[-1]

    else:
        net = models.mobilenet_v3_small(weights=None)

        net.classifier[3] = torch.nn.Linear(
            net.classifier[3].in_features,
            len(CLASSES)
        )

        target_layer = net.features[-1]

    net.load_state_dict(checkpoint["state_dict"])

    net.to(device)
    net.eval()

    model = net

    print(f"Model loaded: {architecture}")
    print(f"Device: {device}")
    print(f"Classes: {CLASSES}")

    return model


# ==================================================
# IMAGE TO BASE64 DATA URL
# ==================================================

def image_to_data_url(image):
    """Convert an image into a format the browser can display."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ==================================================
# GRAD-CAM GENERATION
# ==================================================

def generate_gradcam(net, tensor, class_index):
    """
    Generate a Grad-CAM activation map for a predicted class.

    Grad-CAM visualizes image regions influencing the model output.
    It is not a tumor segmentation mask.
    """

    if target_layer is None:
        raise RuntimeError(
            "Grad-CAM target layer is unavailable."
        )

    activations = []
    gradients = []

    def capture_activation(module, inputs, output):
        activations.append(output)

        output.register_hook(
            lambda gradient: gradients.append(gradient)
        )

    handle = target_layer.register_forward_hook(
        capture_activation
    )

    try:
        net.zero_grad(set_to_none=True)

        with torch.enable_grad():
            input_tensor = (
                tensor.detach()
                .clone()
                .requires_grad_(True)
            )

            logits = net(input_tensor)
            score = logits[0, class_index]

            score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "Grad-CAM activations or gradients are missing."
            )

        feature_maps = activations[0].detach()
        gradient_maps = gradients[0].detach()

        # Calculate the average gradient for each channel.
        weights = gradient_maps.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Combine the feature maps using gradient weights.
        cam = (
            weights * feature_maps
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()

        # Normalize the activation map.
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min > 1e-8:
            cam = (
                (cam - cam_min)
                / (cam_max - cam_min)
            )
        else:
            cam = np.zeros_like(cam)

        heatmap = (
            cam * 255
        ).astype(np.uint8)

        return Image.fromarray(
            heatmap
        ).convert("L")

    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)


# ==================================================
# CREATE GRAD-CAM OVERLAY
# ==================================================

def create_overlay(original_image, heatmap_image):
    """Blend the activation map with the original MRI."""

    original = original_image.resize(
        (224, 224)
    ).convert("RGB")

    heatmap_image = heatmap_image.resize(
        (224, 224)
    ).convert("L")

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    )

    normalized = heatmap_array / 255.0

    # Build a red/yellow activation visualization.
    colored = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    colored[:, :, 0] = np.clip(
        normalized * 255 * 2,
        0,
        255
    ).astype(np.uint8)

    colored[:, :, 1] = np.clip(
        (normalized - 0.35) * 255 * 1.8,
        0,
        255
    ).astype(np.uint8)

    colored[:, :, 2] = 0

    colored_heatmap = Image.fromarray(colored)

    return Image.blend(
        original,
        colored_heatmap,
        alpha=0.40
    )


# ==================================================
# CLASSIFY PREDICTION STATUS
# ==================================================

def get_tumor_status(predicted_class):
    """
    Translate the model's predicted class into a UI status.

    These labels describe model output, not a medical diagnosis.
    """

    if predicted_class == "notumor":
        return {
            "tumor_status": "no_tumor_class_predicted",
            "status_label": "No-Tumor Class Predicted",
            "status_type": "no_tumor"
        }

    return {
        "tumor_status": "tumor_class_predicted",
        "status_label": "Tumor-Class Prediction",
        "status_type": "tumor_predicted"
    }


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        ready=MODEL_PATH.exists()
    )


# ==================================================
# PREDICTION API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "images" not in request.files:
        return jsonify(
            error="Choose at least one image."
        ), 400

    net = model or load_model()

    if net is None:
        return jsonify(
            error=(
                "No trained model found. "
                "Check artifacts/best_model.pt."
            )
        ), 503

    output = []

    for uploaded_file in request.files.getlist("images"):

        if not uploaded_file.filename:
            continue

        try:
            # ------------------------------------------
            # READ ORIGINAL MRI
            # ------------------------------------------

            original_image = Image.open(
                uploaded_file.stream
            ).convert("RGB")

            # ------------------------------------------
            # PREPROCESS IMAGE
            # ------------------------------------------

            tensor = preprocess(
                original_image
            ).unsqueeze(0).to(device)

            # ------------------------------------------
            # PREDICT CLASS
            # ------------------------------------------

            with torch.no_grad():
                logits = net(tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[0].cpu().tolist()

            predicted_index = int(
                max(
                    range(len(probabilities)),
                    key=lambda i: probabilities[i]
                )
            )

            predicted_class = CLASSES[predicted_index]

            confidence = round(
                probabilities[predicted_index] * 100,
                2
            )

            # ------------------------------------------
            # DETERMINE STATUS LABEL
            # ------------------------------------------

            status = get_tumor_status(
                predicted_class
            )

            # ------------------------------------------
            # CREATE ORIGINAL MRI PREVIEW
            # ------------------------------------------

            original_preview = original_image.copy()

            original_preview.thumbnail(
                (512, 512)
            )

            original_data_url = image_to_data_url(
                original_preview
            )

            # ------------------------------------------
            # GENERATE GRAD-CAM
            # ------------------------------------------

            heatmap = generate_gradcam(
                net,
                tensor,
                predicted_index
            )

            overlay = create_overlay(
                original_image,
                heatmap
            )

            heatmap_data_url = image_to_data_url(
                overlay
            )

            # ------------------------------------------
            # CLASS DISTRIBUTION
            # ------------------------------------------

            scores = {
                CLASSES[i]: round(
                    probabilities[i] * 100,
                    2
                )
                for i in range(len(CLASSES))
            }

            # ------------------------------------------
            # RETURN COMPLETE RESULT
            # ------------------------------------------

            output.append({
                "filename": uploaded_file.filename,

                "prediction": predicted_class,

                "confidence": confidence,

                "tumor_status": status["tumor_status"],

                "status_label": status["status_label"],

                "status_type": status["status_type"],

                "scores": scores,

                "original_image": original_data_url,

                "heatmap_image": heatmap_data_url
            })

        except Exception:
            app.logger.exception(
                "Failed to process uploaded MRI image."
            )

            output.append({
                "filename": uploaded_file.filename,
                "error": (
                    "Image processing failed. "
                    "Check the Flask terminal for details."
                )
            })

    return jsonify(
        results=output,
        disclaimer=(
            "Educational prediction only. "
            "This system is not a medical diagnostic tool. "
            "A no-tumor-class prediction does not rule out disease."
        )
    )


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    if MODEL_PATH.exists():
        load_model()
    else:
        print(
            "WARNING: Model checkpoint not found:",
            MODEL_PATH
        )

    app.run(debug=True)

