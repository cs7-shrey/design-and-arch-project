"""Generate measured figures and a presentation brief from saved evaluation files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.prepare import ROOT

NAMES={'baseline':'Compact U-Net','softroi':'SoftROI ablation','srb':'SRB-U-Net'}
COLORS={'baseline':'#718096','softroi':'#E7A341','srb':'#008879'}
CLASSES=['Meningioma','Glioma','Pituitary']


def generate(run):
    run=ROOT/run
    out=ROOT/'output';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    results=json.loads((run/'results.json').read_text())
    data=np.load(ROOT/'data/processed/subset.npz')
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    records=[r for r in manifest['records'] if r['split']=='test']
    ids=np.flatnonzero(data['splits']=='test')
    images,masks,labels=data['images'][ids],data['masks'][ids],data['labels'][ids]
    cover_index=int(np.flatnonzero(data['splits']=='train')[0])
    Image.fromarray((data['images'][cover_index]*255).astype(np.uint8)).resize((768,768)).save(figs/'cover_mri.png')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for variant in NAMES:
        histories=[]
        for seed in results['configuration']['seeds']:
            with (run/f'{variant}_seed{seed}'/'history.csv').open() as f: histories.append(list(csv.DictReader(f)))
        curves=np.array([[float(r['val_patient_dice']) for r in h] for h in histories])
        x=np.arange(1,curves.shape[1]+1)
        axes[0].plot(x,curves.mean(0)*100,label=NAMES[variant],color=COLORS[variant],lw=2)
        axes[0].fill_between(x,curves.min(0)*100,curves.max(0)*100,color=COLORS[variant],alpha=.12)
    axes[0].set(xlabel='Epoch',ylabel='Validation patient Dice (%)',title='Learning curves (band spans the two seeds)')
    axes[0].legend(fontsize=9)
    keys=['patient_dice','patient_iou','slice_boundary_f1']
    x=np.arange(3)
    for j,variant in enumerate(NAMES):
        a=results['aggregate'][variant]
        axes[1].bar(x+(j-1)*.25,[a['means'][k]*100 for k in keys],.24,label=NAMES[variant],color=COLORS[variant])
    axes[1].set(xticks=x,xticklabels=['Patient Dice','Patient IoU','Boundary F1'],ylabel='Held-out score (%)',ylim=(0,100),title='Test results averaged over two seeds')
    axes[1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(figs/'learning_and_results.png',dpi=180);plt.close(fig)
    # Fix the qualitative seed in advance and choose the first manifest slice of each class.
    seed=results['configuration']['seeds'][0]
    base=np.load(run/f'baseline_seed{seed}'/'test_predictions.npz')
    srb=np.load(run/f'srb_seed{seed}'/'test_predictions.npz')
    chosen=[int(np.flatnonzero(labels==k)[0]) for k in range(3)]
    def overlay(im,mask,color):
        rgb=np.repeat(im[:,:,None],3,axis=2)
        m=mask.astype(bool);rgb[m]=rgb[m]*.55+np.array(color)*.45
        return np.clip(rgb,0,1)
    fig,axes=plt.subplots(3,4,figsize=(11,8.7))
    for row,i in enumerate(chosen):
        views=[images[i],overlay(images[i],masks[i],[.15,.9,.65]),overlay(images[i],base['probabilities'][i]>=.5,[1,.6,.2]),overlay(images[i],srb['probabilities'][i]>=.5,[.15,.75,1])]
        for col,view in enumerate(views):
            axes[row,col].imshow(view,cmap='gray',vmin=0,vmax=1);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(['MRI','Ground truth','Compact U-Net','SRB-U-Net'][col])
            if col==0: axes[row,col].set_ylabel(f'{CLASSES[row]}\n{records[i]["file"]}',fontsize=11)
        for name,array in [('mri',images[i]),('truth',views[1]),('baseline',views[2]),('srb',views[3])]:
            Image.fromarray((array*255).astype(np.uint8)).resize((512,512)).save(figs/f'example_{row}_{name}.png')
    fig.suptitle(f'First held-out slice of each class in frozen manifest (seed {seed})',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(figs/'qualitative_examples.png',dpi=180);plt.close(fig)
    # Show the worst relative case, explicitly labelled, instead of only favorable cases.
    with (run/'test_per_slice.csv').open() as f: rows=list(csv.DictReader(f))
    b={r['file']:float(r['dice']) for r in rows if r['variant']=='baseline' and int(r['seed'])==seed}
    s={r['file']:float(r['dice']) for r in rows if r['variant']=='srb' and int(r['seed'])==seed}
    worst=min(range(len(records)),key=lambda i:s[records[i]['file']]-b[records[i]['file']])
    fig,axes=plt.subplots(1,4,figsize=(11,3))
    views=[images[worst],overlay(images[worst],masks[worst],[.15,.9,.65]),overlay(images[worst],base['probabilities'][worst]>=.5,[1,.6,.2]),overlay(images[worst],srb['probabilities'][worst]>=.5,[.15,.75,1])]
    for ax,v,title in zip(axes,views,['MRI','Ground truth',f"Baseline Dice {b[records[worst]['file']]*100:.1f}%",f"SRB Dice {s[records[worst]['file']]*100:.1f}%"]):
        ax.imshow(v,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f'Largest SRB regression relative to baseline: {records[worst]["file"]} (seed {seed})',fontsize=12)
    fig.tight_layout();fig.savefig(figs/'failure_case.png',dpi=180);plt.close(fig)
    # Mechanism visualization is an observation, not a calibrated explanation.
    i=chosen[1]
    fig,axes=plt.subplots(1,4,figsize=(11,3.1))
    for ax,v,title in zip(axes,[images[i],srb['coarse_probabilities'][i],srb['boundary_probabilities'][i],srb['probabilities'][i]],['MRI','Predicted soft ROI','Predicted boundary','Final tumor probability']):
        ax.imshow(v,cmap='gray' if title=='MRI' else 'magma',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(figs/'mechanism_maps.png',dpi=180);plt.close(fig)
    delta=results['srb_vs_baseline'];agg=results['aggregate']
    table=['| Model | Parameters | Patient Dice | Patient IoU | Boundary F1 |','|---|---:|---:|---:|---:|']
    for variant in NAMES:
        a=agg[variant];m=a['means']
        table.append(f"| {NAMES[variant]} | {a['parameters']:,} | {m['patient_dice']*100:.2f}% | {m['patient_iou']*100:.2f}% | {m['slice_boundary_f1']*100:.2f}% |")
    lo,hi=delta['paired_patient_bootstrap_95_ci']
    md=f'''# Tomorrow's project update

## Proposed work

**SoftROI-Boundary U-Net (SRB-U-Net)** learns where the tumor may be without cropping the MRI, then uses predicted boundaries to refine uncertain segmentation pixels. This is an implemented project proposal built from established research ideas. Global novelty has not been established.

## What actually ran

398 training slices from 48 patients, 117 validation slices from 12 patients, and 129 test slices from 15 patients. Three variants, two seeds, 60 epochs per run, 128 x 128 inputs, random initialization, Apple M4 GPU. Every variant shares the U-Net backbone, auxiliary class head, training budget, and data. Checkpoints use validation patient Dice, with a fixed 0.5 threshold. The 20-epoch validation pilot prompted the longer budget before any test scoring.

## Measured test results

{chr(10).join(table)}

Values are arithmetic averages of the two independently trained seeds. Patient Dice/IoU first average slices within each patient. Boundary F1 uses a two-pixel tolerance at 128 x 128 and averages slices.

SRB versus baseline: **{delta['patient_dice_delta']*100:+.2f} percentage points in patient Dice**. Paired patient bootstrap 95% interval: **[{lo*100:+.2f}, {hi*100:+.2f}] points**. {delta['patients_improved']}/15 patients improved after averaging the two seeds. This interval reflects the held-out patients, not all uncertainty from training or choosing a different split.

The full proposal has {agg['srb']['parameters']:,} parameters, versus {agg['baseline']['parameters']:,} in the baseline, an increase of {(agg['srb']['parameters']/agg['baseline']['parameters']-1)*100:.2f}%.

## Suggested 60-second update

“I identified a limitation in the base paper's detection-then-crop pipeline: a bounding box can restrict the context available to the segmenter. I implemented SoftROI-Boundary U-Net, which predicts a soft tumor region inside one network and uses it to guide skip features while retaining the full image. A boundary branch helps a residual head refine pixels near uncertain predictions.

I tested the proposal against a compact U-Net and a region-only ablation using a patient-separated subset. Across two seeds, baseline patient Dice was {agg['baseline']['means']['patient_dice']*100:.2f}% and the full model reached {agg['srb']['means']['patient_dice']*100:.2f}%. The region-only ablation reached {agg['softroi']['means']['patient_dice']*100:.2f}%. These are preliminary measurements. Next I will test more patient splits and separate the effects of boundary supervision and refinement.”

## Likely questions

**What is new?** The proposed integration: a supervised coarse ROI gate with a context floor, plus boundary-conditioned refinement weighted by the provisional mask's uncertainty proxy. The individual ideas have prior work. Describe this as a project contribution pending a fuller novelty review.

**Is this the paper's YOLOv5 + U-Net implementation?** No. It is a compact alternative tested against an internal controlled baseline. The paper's 88.1% Dice and 89.5% detection mAP are reference results under different conditions and are not directly comparable.

**Is this tumor detection?** It localizes a tumor using a predicted mask and can derive a bounding box from that mask. The auxiliary head predicts one of the three tumor types. This run does not evaluate object detection mAP or healthy-versus-tumor screening.

**Did masks leak into inference?** No. Forward inference accepts MRI intensity only. Masks supply training targets and held-out scoring. Patient IDs keep all slices from a patient in one split.

**Why compare an ablation?** SoftROI versus baseline tests region supervision/gating together. Full SRB versus SoftROI tests the added boundary/refinement package. Further experiments must separate those components and their losses.

**Why not report pixel accuracy?** Most pixels are background, so it can look high while missing tumors. Dice/IoU and boundary F1 measure the target more directly.

**Does this prove a robust improvement?** No. Only 15 test patients, two seeds, one split, and lower-resolution inputs. Read the paired interval and seed-specific numbers, including regressions.

## Evidence and reproducibility

- `runs/main/results.json`: complete metrics, per-class scores, bootstrap interval, all seeds.
- `runs/main/metrics.csv` and `test_per_slice.csv`: auditable numerical results.
- `runs/main/*/history.csv`: per-epoch training and validation logs.
- `runs/main/*/checkpoints/best.pt`: all six validation-selected trained models.
- `data/processed/manifest.json`: fixed subset and patient splits.
- `output/figures/qualitative_examples.png`: first test slice per class in manifest order, seed 17.
- `output/figures/failure_case.png`: worst SRB-versus-baseline slice for seed 17.
- `EXPERIMENT_PROTOCOL.md`: methodology and prior-work links.

## Next experiments

1. Repeat on all patients with multiple patient-group folds and additional seeds.
2. Separate coarse supervision, skip gating, boundary supervision, and residual refinement in a fuller ablation.
3. Increase resolution to 224/256, assess tiny tumors, and compare a faithful YOLOv5 + U-Net baseline.
4. Add normal MRI data before attempting tumor-presence screening.
'''
    (out/'PRESENTATION_BRIEF.md').write_text(md)
    (out/'qualitative_selection.json').write_text(json.dumps({'seed':seed,'fixed_examples':[records[i]['file'] for i in chosen],'selection_rule':'first test slice per class in frozen manifest','worst_relative_case':records[worst]['file']},indent=2))
    print(out/'PRESENTATION_BRIEF.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/main');a=p.parse_args();generate(a.run)
