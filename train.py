import argparse
import random
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model(name):
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 4)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 4)
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    return m

def get_target_layer(model, name):
    """Returns the target convolutional layer for Grad-CAM."""
    if name == "resnet18":
        return model.layer4[-1]
    elif name in ["mobilenet_v3_small", "efficientnet_b0"]:
        return model.features[-1]
    raise ValueError(f"Unknown model name: {name}")

class GradCAM:
    """Dependency-free PyTorch implementation of Grad-CAM."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        gradients = self.gradients.data[0]
        activations = self.activations.data[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * activations, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        prob = torch.softmax(logits, dim=1)[0, class_idx].item()
        return cam, class_idx, prob

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

def save_confusion_matrix_heatmap(cm, classes, save_path, title="Confusion Matrix"):
    """Plots and saves a confusion matrix seaborn heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_gradcam_overlay(img_tensor, cam, true_class, pred_class, prob, save_path):
    """Overlays Grad-CAM heatmap on the original MRI scan image and saves it."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # Resize CAM to image resolution (224, 224)
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = torch.nn.functional.interpolate(
        cam_tensor, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_np)
    ax[0].set_title(f"Original MRI (True: {true_class})")
    ax[0].axis('off')

    ax[1].imshow(img_np)
    ax[1].imshow(cam_resized, cmap='jet', alpha=0.45)
    ax[1].set_title(f"Grad-CAM (Pred: {pred_class} | {prob*100:.1f}%)")
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(ys, ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="Root containing Training/ and Testing/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-cam-samples", type=int, default=4, help="Number of test Grad-CAM samples to generate per model")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root = Path(args.data)
    train_dir, test_dir = root / "Training", root / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit("Expected dataset/Training and dataset/Testing.")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    if train_aug.classes != CLASSES or test_ds.classes != CLASSES:
        raise SystemExit(f"Unexpected class folders. Expected {CLASSES}; got train={train_aug.classes}, test={test_ds.classes}")

    indices = np.random.permutation(len(train_aug)).tolist()
    n_val = max(1, int(0.15 * len(indices)))
    val_idx, fit_idx = indices[:n_val], indices[n_val:]

    train_loader = DataLoader(Subset(train_aug, fit_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(train_eval, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out = Path("artifacts")
    heatmap_out = out / "heatmaps"
    out.mkdir(exist_ok=True)
    heatmap_out.mkdir(exist_ok=True)

    summary = {}
    overall_best_val_accuracy = -1.0

    for name in ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]:
        print(f"\n=== Training {name} on {DEVICE} ===")
        model = make_model(name).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        best_acc = -1
        best_state = None

        for epoch in range(args.epochs):
            tl, ta = run_epoch(model, train_loader, optimizer)
            vl, va = run_epoch(model, val_loader)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")
            if va > best_acc:
                best_acc = va
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        
        # Test Evaluation
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(DEVICE)).argmax(1).cpu().tolist()
                ys.extend(y.tolist())
                ps.extend(pred)

        report = classification_report(ys, ps, target_names=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(ys, ps, labels=list(range(4))).tolist()
        acc = float(accuracy_score(ys, ps))
        summary[name] = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}

        # 1. Save Confusion Matrix Heatmap Image
        cm_path = heatmap_out / f"cm_{name}.png"
        save_confusion_matrix_heatmap(cm, CLASSES, cm_path, title=f"Confusion Matrix - {name}")
        print(f"Saved confusion matrix heatmap to {cm_path}")

        # 2. Generate and Save Grad-CAM Attention Heatmaps
        print(f"Generating Grad-CAM heatmaps for {name}...")
        target_layer = get_target_layer(model, name)
        grad_cam = GradCAM(model, target_layer)

        for sample_idx in range(min(args.num_cam_samples, len(test_ds))):
            img_tensor, true_label = test_ds[sample_idx]
            x_input = img_tensor.unsqueeze(0).to(DEVICE)

            cam, pred_label, prob = grad_cam.generate(x_input)
            grad_cam_path = heatmap_out / f"gradcam_{name}_sample_{sample_idx+1}.png"
            save_gradcam_overlay(
                img_tensor, cam, CLASSES[true_label], CLASSES[pred_label], prob, grad_cam_path
            )

        grad_cam.remove_hooks()

        # Save checkpoint
        torch.save({"architecture": name, "state_dict": model.state_dict(), "classes": CLASSES}, out / f"{name}.pt")

        if best_acc > overall_best_val_accuracy:
            overall_best_val_accuracy = best_acc
            torch.save(
                {
                    "architecture": name,
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "validation_accuracy": best_acc,
                },
                out / "best_model.pt",
            )

    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nResults saved to artifacts/metrics.json")
    print("Confusion matrices & Grad-CAM overlays saved to artifacts/heatmaps/")

if __name__ == "__main__":
    main()