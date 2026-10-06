"""Independently audit headline scores against saved probability arrays."""
import csv
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
run=root/'runs/main'
report=json.loads((run/'results.json').read_text())
data=np.load(root/'data/processed/subset.npz')
truth=data['masks'][data['splits']=='test'].astype(bool)
manifest=json.loads((root/'data/processed/manifest.json').read_text())
records=[r for r in manifest['records'] if r['split']=='test']
pids=np.array([r['pid'] for r in records])
assert len(np.unique(pids))==15
assert len(report['individual_runs'])==6
for result in report['individual_runs']:
    folder=run/f"{result['variant']}_seed{result['seed']}"
    prediction=np.load(folder/'test_predictions.npz')['probabilities']
    assert prediction.shape==truth.shape
    assert np.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=1
    pred=prediction>=.5
    intersection=(pred & truth).sum((1,2))
    dice=(2*intersection+1e-7)/(pred.sum((1,2))+truth.sum((1,2))+1e-7)
    patient_dice=np.mean([dice[pids==pid].mean() for pid in np.unique(pids)])
    assert np.isclose(dice.mean(),result['slice_dice'],atol=1e-6)
    assert np.isclose(patient_dice,result['patient_dice'],atol=1e-6)
    with (folder/'history.csv').open() as f: history=list(csv.DictReader(f))
    best=max(history,key=lambda r:float(r['val_patient_dice']))
    assert int(best['epoch'])==result['best_epoch']
    assert len(history)==60
    print(folder.name, 'prediction-derived Dice and checkpoint selection verified')
for variant,aggregate in report['aggregate'].items():
    seeds=[r for r in report['individual_runs'] if r['variant']==variant]
    for key,value in aggregate['means'].items():
        assert np.isclose(value,np.mean([r[key] for r in seeds]),atol=1e-10)
print('All headline metrics, seed averages, and checkpoint choices verified.')
