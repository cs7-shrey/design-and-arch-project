"""Freeze a deterministic, patient-disjoint subset before training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def prepare(size=128, seed=2026, train_patients=16, val_patients=4, test_patients=5, max_slices=10):
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    patients = defaultdict(list)
    with h5py.File(ROOT / "data/cvind.mat") as f:
        folds = f["cvind"][()].ravel().astype(int)
    for p in sorted((ROOT / "data/mat").glob("*.mat"), key=lambda p: int(p.stem)):
        with h5py.File(p) as f:
            g = f["cjdata"]
            label = int(g["label"][()].item()) - 1
            pid = "".join(chr(int(v)) for v in g["PID"][()].ravel())
        patients[pid].append({"file": p.name, "label": label, "pid": pid, "fold": int(folds[int(p.stem)-1])})
    assert all(len({r["fold"] for r in rows}) == 1 for rows in patients.values())
    rng = np.random.default_rng(seed)
    records = []
    for split, allowed, count in [("train", {1,2,3}, train_patients), ("val", {4}, val_patients), ("test", {5}, test_patients)]:
        for label in range(3):
            candidates = sorted(pid for pid, rows in patients.items() if rows[0]["label"] == label and rows[0]["fold"] in allowed)
            chosen = rng.choice(candidates, size=count, replace=False)
            for pid in sorted(chosen):
                rows = patients[pid]
                selected = sorted(rng.choice(len(rows), size=min(max_slices, len(rows)), replace=False))
                records.extend(dict(rows[i], split=split) for i in selected)
    images, masks = [], []
    for r in records:
        with h5py.File(ROOT / "data/mat" / r["file"]) as f:
            # h5py reads MATLAB arrays with reversed dimensions. Transpose BOTH.
            im = f["cjdata/image"][()].T.astype(np.float32)
            mask = f["cjdata/tumorMask"][()].T.astype(np.float32)
        r["original_shape"] = list(im.shape)
        # Per-image statistics use MRI only; no mask-dependent preprocessing.
        hi = float(np.percentile(im[im > 0], 99)) if np.any(im > 0) else 1.0
        im = np.clip(im / max(hi, 1), 0, 1)
        im = F.interpolate(torch.from_numpy(im)[None,None], size=(size,size), mode="bilinear", align_corners=False)[0,0].numpy()
        mask = F.interpolate(torch.from_numpy(mask)[None,None], size=(size,size), mode="nearest")[0,0].numpy()
        if not mask.any():
            raise ValueError(f"Resizing erased mask for {r['file']}")
        images.append(im)
        masks.append(mask.astype(np.uint8))
    sets = {s: {r["pid"] for r in records if r["split"] == s} for s in ["train", "val", "test"]}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    summary = {s: {"patients": len(sets[s]), "slices": sum(r["split"] == s for r in records), "class_counts": dict(Counter(r["label"] for r in records if r["split"] == s))} for s in sets}
    manifest = {"seed": seed, "size": size, "max_slices_per_patient": max_slices, "original_folds": {"train": [1,2,3], "val": [4], "test": [5]}, "summary": summary, "records": records}
    raw = json.dumps(manifest, indent=2)
    (out / "manifest.json").write_text(raw)
    np.savez_compressed(out / "subset.npz", images=np.stack(images), masks=np.stack(masks), labels=np.array([r["label"] for r in records]), splits=np.array([r["split"] for r in records]))
    print(json.dumps({"summary": summary, "manifest_sha256": hashlib.sha256(raw.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()
    prepare(size=args.size)
