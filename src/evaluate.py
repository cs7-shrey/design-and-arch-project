"""One final held-out evaluation of validation-selected checkpoints."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from src.models import TumorNet
from src.prepare import ROOT
from src.train import metrics_per_slice

CLASSES = ["meningioma", "glioma", "pituitary"]


def boundary_f1(pred, target, tolerance=2):
    pred_edge = pred ^ ndimage.binary_erosion(pred)
    target_edge = target ^ ndimage.binary_erosion(target)
    if not pred_edge.any(): return 0.0
    precision = float((ndimage.distance_transform_edt(~target_edge)[pred_edge] <= tolerance).mean())
    recall = float((ndimage.distance_transform_edt(~pred_edge)[target_edge] <= tolerance).mean())
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def patient_mean(rows, key):
    groups = defaultdict(list)
    for row in rows: groups[row['pid']].append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()]))


@torch.no_grad()
def evaluate(root):
    torch.set_num_threads(4)
    root = ROOT / root
    config = json.loads((root / 'config.json').read_text())
    expected = [root / f'{v}_seed{s}' for v,s in config['jobs']]
    if not all((p/'complete.json').exists() for p in expected):
        raise RuntimeError('All planned training jobs must complete before test evaluation.')
    manifest_text = (ROOT/'data/processed/manifest.json').read_text()
    assert hashlib.sha256(manifest_text.encode()).hexdigest() == config['manifest_sha256']
    manifest = json.loads(manifest_text)
    records = [r for r in manifest['records'] if r['split']=='test']
    data = np.load(ROOT/'data/processed/subset.npz')
    ids = np.flatnonzero(data['splits']=='test')
    x = torch.from_numpy(data['images'][ids,None].copy())
    y = torch.from_numpy(data['masks'][ids,None].astype(np.float32))
    labels = data['labels'][ids]
    train_areas = data['masks'][data['splits']=='train'].mean((1,2))
    small_cutoff = float(np.median(train_areas))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    results, all_rows = [], []
    for path in expected:
        ckpt = torch.load(path/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        model = TumorNet(ckpt['variant'],ckpt['width']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        seg, cls, coarse, boundary = [], [], [], []
        for batch in x.split(16):
            out = model(batch.to(device))
            seg.append(out['seg_logits'].cpu())
            cls.append(out['class_logits'].softmax(1).cpu())
            for key, dest in [('coarse_logits',coarse),('boundary_logits',boundary)]:
                if key in out:
                    dest.append(torch.nn.functional.interpolate(out[key].sigmoid(),size=x.shape[-2:],mode='bilinear',align_corners=False).cpu())
        logits = torch.cat(seg)
        probs = logits.sigmoid().numpy()[:,0]
        class_probs = torch.cat(cls).numpy()
        metrics = {k:v.numpy() for k,v in metrics_per_slice(logits,y).items()}
        rows = []
        for i,r in enumerate(records):
            rows.append({**{k:r[k] for k in ['file','pid','label']},'variant':ckpt['variant'],'seed':ckpt['seed'],**{k:float(v[i]) for k,v in metrics.items()},'boundary_f1':boundary_f1(probs[i]>=0.5,y[i,0].numpy()>0.5),'area_fraction':float(y[i].mean()),'class_pred':int(class_probs[i].argmax())})
        summary = json.loads((path/'complete.json').read_text())
        summary.update({f'slice_{k}':float(np.mean([r[k] for r in rows])) for k in ['dice','iou','precision','recall','boundary_f1']})
        summary['patient_dice'] = patient_mean(rows,'dice')
        summary['patient_iou'] = patient_mean(rows,'iou')
        summary['class_accuracy'] = float(accuracy_score(labels,class_probs.argmax(1)))
        summary['class_macro_f1'] = float(f1_score(labels,class_probs.argmax(1),labels=[0,1,2],average='macro',zero_division=0))
        summary['confusion_matrix'] = confusion_matrix(labels,class_probs.argmax(1),labels=[0,1,2]).tolist()
        summary['per_class_patient_dice'] = {name:patient_mean([r for r in rows if r['label']==i],'dice') for i,name in enumerate(CLASSES)}
        summary['small_tumor_slice_dice'] = float(np.mean([r['dice'] for r in rows if r['area_fraction']<=small_cutoff]))
        summary['large_tumor_slice_dice'] = float(np.mean([r['dice'] for r in rows if r['area_fraction']>small_cutoff]))
        np.savez_compressed(path/'test_predictions.npz',probabilities=probs,class_probabilities=class_probs,coarse_probabilities=torch.cat(coarse).numpy()[:,0] if coarse else np.empty(0),boundary_probabilities=torch.cat(boundary).numpy()[:,0] if boundary else np.empty(0))
        (path/'test_metrics.json').write_text(json.dumps(summary,indent=2))
        results.append(summary)
        all_rows.extend(rows)
        print(f"{path.name}: patient Dice={summary['patient_dice']:.4f}, IoU={summary['patient_iou']:.4f}, boundary F1={summary['slice_boundary_f1']:.4f}",flush=True)
        del model
    aggregate = {}
    keys = ['patient_dice','patient_iou','slice_dice','slice_iou','slice_precision','slice_recall','slice_boundary_f1','class_accuracy','class_macro_f1','small_tumor_slice_dice','large_tumor_slice_dice']
    for variant in config['variants']:
        variant_results = [r for r in results if r['variant']==variant]
        aggregate[variant] = {'parameters':variant_results[0]['parameters'],'n_seeds':len(variant_results),'means':{k:float(np.mean([r[k] for r in variant_results])) for k in keys},'seed_std':{k:float(np.std([r[k] for r in variant_results],ddof=1)) if len(variant_results)>1 else None for k in keys},'per_class_patient_dice':{c:float(np.mean([r['per_class_patient_dice'][c] for r in variant_results])) for c in CLASSES}}
    paired = {}
    pids = sorted({r['pid'] for r in records})
    for variant in config['variants']:
        by_patient = {pid:float(np.mean([r['dice'] for r in all_rows if r['variant']==variant and r['pid']==pid])) for pid in pids}
        paired[variant] = by_patient
    comparison = {}
    if 'baseline' in paired and 'srb' in paired:
        delta = np.array([paired['srb'][p]-paired['baseline'][p] for p in pids])
        rng = np.random.default_rng(4242)
        bootstrap = rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
        comparison = {'patient_dice_delta':float(delta.mean()),'paired_patient_bootstrap_95_ci':np.quantile(bootstrap,[.025,.975]).tolist(),'n_patients':len(pids),'patients_improved':int((delta>0).sum()),'method':'10,000 paired patient bootstrap resamples, seeds averaged within patients; does not capture full training or split variability'}
    report = {'configuration':config,'subset_summary':manifest['summary'],'small_tumor_cutoff_train_median':small_cutoff,'small_test_slices':sum(float(v.mean())<=small_cutoff for v in y),'individual_runs':results,'aggregate':aggregate,'srb_vs_baseline':comparison,'patient_scores':paired}
    (root/'results.json').write_text(json.dumps(report,indent=2))
    with (root/'test_per_slice.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    with (root/'metrics.csv').open('w',newline='') as f:
        fields=['variant','seed','parameters','best_epoch',*keys]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/pilot');args=p.parse_args();evaluate(args.run)
