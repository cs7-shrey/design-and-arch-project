"""Train all planned comparisons using validation data only; no test evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from src.models import TumorNet, objective
from src.prepare import ROOT


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_per_slice(out, mask):
    pred = out.sigmoid() >= 0.5
    target = mask > 0.5
    dims = (1,2,3)
    tp = (pred & target).sum(dims).float()
    pp, tt = pred.sum(dims).float(), target.sum(dims).float()
    return {"dice": (2*tp+1e-7)/(pp+tt+1e-7), "iou": (tp+1e-7)/(pp+tt-tp+1e-7), "precision": tp/pp.clamp_min(1), "recall": tp/tt.clamp_min(1)}


def augment(x, y, generator):
    # Same augmentation stream for every variant sharing a seed.
    if torch.rand((), generator=generator) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    k = int(torch.randint(0, 4, (), generator=generator))
    x, y = torch.rot90(x,k,(-2,-1)), torch.rot90(y,k,(-2,-1))
    gain = float(0.9 + 0.2 * torch.rand((), generator=generator))
    return (x*gain).clamp(0,1), y


@torch.no_grad()
def validate(model, x, y, labels, batch_size, device, pids):
    model.eval()
    dice, correct = [], []
    for start in range(0, len(x), batch_size):
        out = model(x[start:start+batch_size].to(device))
        dice.extend(metrics_per_slice(out["seg_logits"], y[start:start+batch_size].to(device))["dice"].cpu().tolist())
        correct.extend((out["class_logits"].argmax(1).cpu() == labels[start:start+batch_size]).tolist())
    patient_scores = [np.mean([d for d,p in zip(dice,pids) if p == pid]) for pid in sorted(set(pids))]
    return float(np.mean(patient_scores)), float(np.mean(dice)), float(np.mean(correct))


def run(args):
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / "data/processed/subset.npz")
    manifest_text = (ROOT / "data/processed/manifest.json").read_text()
    manifest = json.loads(manifest_text)
    parts = {}
    for split in ["train", "val"]:
        ii = np.flatnonzero(data["splits"] == split)
        parts[split] = (torch.from_numpy(data["images"][ii,None].copy()), torch.from_numpy(data["masks"][ii,None].astype(np.float32)), torch.from_numpy(data["labels"][ii].copy()).long())
    val_pids = [r["pid"] for r in manifest["records"] if r["split"] == "val"]
    jobs = [(variant, seed) for seed in args.seeds for variant in args.variants]
    config = dict(vars(args), input_size=int(data['images'].shape[-1]), device=str(device), torch_version=torch.__version__, platform=platform.platform(), manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(), jobs=jobs, selection="maximum validation patient-mean Dice at fixed threshold 0.5", training_loss="0.5 weighted BCE + 0.5 Dice + 0.15 CE; softroi adds 0.25 coarse loss; srb adds 0.10 boundary loss")
    if (root / "config.json").exists():
        old = json.loads((root / "config.json").read_text())
        for key in ["variants", "seeds", "epochs", "batch_size", "width", "lr", "manifest_sha256"]:
            if old[key] != config[key]:
                raise ValueError(f"Existing run has different {key}; choose a new --output directory.")
    (root / "config.json").write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    for variant, seed in jobs:
        path = root / f"{variant}_seed{seed}"
        if (path / "complete.json").exists():
            print(f"Already complete: {path.name}", flush=True)
            continue
        path.mkdir(exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)
        seed_all(seed)
        shared = TumorNet("baseline", args.width).state_dict()
        seed_all(seed)
        model = TumorNet(variant,args.width)
        model.load_state_dict(shared, strict=False)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*0.1)
        g = torch.Generator().manual_seed(seed + 10000)
        history = []
        best, best_epoch = -1, 0
        start_run = time.perf_counter()
        x,y,labels = parts["train"]
        for epoch in range(1,args.epochs+1):
            start = time.perf_counter()
            model.train()
            order = torch.randperm(len(x),generator=g)
            loss_sum = torch.zeros((),device=device)
            for ids in order.split(args.batch_size):
                xb,yb = augment(x[ids],y[ids],g)
                xb,yb,lb = xb.to(device),yb.to(device),labels[ids].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(xb),yb,lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
                loss_sum += loss.detach()*len(ids)
            val_patient,val_dice,val_acc = validate(model,*parts["val"],args.batch_size,device,val_pids)
            row = {"epoch":epoch,"train_loss":float(loss_sum.cpu())/len(x),"val_patient_dice":val_patient,"val_slice_dice":val_dice,"val_class_accuracy":val_acc,"lr":optimizer.param_groups[0]["lr"],"seconds":time.perf_counter()-start}
            scheduler.step()
            history.append(row)
            if val_patient > best:
                best,best_epoch = val_patient,epoch
                state = {"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"variant":variant,"width":args.width,"seed":seed,"epoch":epoch,"validation_patient_dice":best,"config":config}
                torch.save(state,path / "checkpoints/best.pt")
            with (path / "history.csv").open("w",newline="") as f:
                writer = csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(f"{variant:8s} seed={seed} epoch={epoch:02d}/{args.epochs} loss={row['train_loss']:.4f} val_patient_Dice={val_patient:.4f} val_slice_Dice={val_dice:.4f} seconds={row['seconds']:.1f}",flush=True)
        result = {"variant":variant,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),"best_epoch":best_epoch,"validation_patient_dice":best,"training_seconds":time.perf_counter()-start_run}
        (path / "complete.json").write_text(json.dumps(result,indent=2))
        del model,optimizer
        if device.type == "mps": torch.mps.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants",nargs="+",default=["baseline","softroi","srb"])
    p.add_argument("--seeds",nargs="+",type=int,default=[17,29])
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--width",type=int,default=16)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--output",default="runs/pilot")
    run(p.parse_args())
