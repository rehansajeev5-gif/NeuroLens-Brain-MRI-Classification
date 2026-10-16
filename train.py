import argparse, json, copy, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, ys, ps = 0.0, [], []
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits=model(x); loss=loss_fn(logits,y)
            if training: loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
        ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist())
    return total_loss/len(loader.dataset), accuracy_score(ys,ps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="Root containing Training/ and Testing/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(args.data)
    train_dir, test_dir=root/"Training", root/"Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit("Expected dataset/Training and dataset/Testing. Extract archive.zip as described in README.md.")
    train_tf=transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_aug=datasets.ImageFolder(train_dir, transform=train_tf)
    train_eval=datasets.ImageFolder(train_dir, transform=eval_tf)
    test_ds=datasets.ImageFolder(test_dir, transform=eval_tf)
    if train_aug.classes != CLASSES or test_ds.classes != CLASSES:
        raise SystemExit(f"Unexpected class folders. Expected {CLASSES}; got train={train_aug.classes}, test={test_ds.classes}")
    # Hold out 15% of Training for model selection; reserve Testing for final evaluation only.
    indices=np.random.permutation(len(train_aug)).tolist()
    n_val=max(1,int(0.15*len(indices)))
    val_idx, fit_idx=indices[:n_val], indices[n_val:]
    train_loader=DataLoader(Subset(train_aug,fit_idx),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(Subset(train_eval,val_idx),batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    out=Path("artifacts"); out.mkdir(exist_ok=True)
    summary={}
    for name in ["mobilenet_v3_small","resnet18","efficientnet_b0"]:
        print(f"\n=== Training {name} on {DEVICE} ===")
        model=make_model(name).to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        best_acc=-1; best_state=None
        for epoch in range(args.epochs):
            tl,ta=run_epoch(model,train_loader,optimizer)
            vl,va=run_epoch(model,val_loader)
            print(f"epoch {epoch+1}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} test_loss={vl:.4f} test_acc={va:.4f}")
            if va > best_acc:
                best_acc=va; best_state=copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        ys=[]; ps=[]
        model.eval()
        with torch.no_grad():
            for x,y in test_loader:
                pred=model(x.to(DEVICE)).argmax(1).cpu().tolist()
                ys.extend(y.tolist()); ps.extend(pred)
        report=classification_report(ys,ps,target_names=CLASSES,output_dict=True,zero_division=0)
        cm=confusion_matrix(ys,ps,labels=list(range(4))).tolist()
        summary[name]={"accuracy":float(accuracy_score(ys,ps)),"classification_report":report,"confusion_matrix":cm}
        torch.save({"architecture":name,"state_dict":model.state_dict(),"classes":CLASSES},out/f"{name}.pt")
        if summary[name]["accuracy"] > summary.get("_best_accuracy", -1):
            summary["_best_accuracy"]=summary[name]["accuracy"]
            torch.save({"architecture":name,"state_dict":model.state_dict(),"classes":CLASSES},out/"best_model.pt")
    (out/"metrics.json").write_text(json.dumps(summary,indent=2))
    print("\nResults saved to artifacts/metrics.json; best model to artifacts/best_model.pt")
    print("Reminder: dataset test accuracy is not evidence of clinical reliability.")

if __name__=="__main__": main()
