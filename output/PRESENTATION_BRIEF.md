# Tomorrow's project update

## Proposed work

**SoftROI-Boundary U-Net (SRB-U-Net)** learns where the tumor may be without cropping the MRI, then uses predicted boundaries to refine uncertain segmentation pixels. This is an implemented project proposal built from established research ideas. Global novelty has not been established.

## What actually ran

398 training slices from 48 patients, 117 validation slices from 12 patients, and 129 test slices from 15 patients. Three variants, two seeds, 60 epochs per run, 128 x 128 inputs, random initialization, Apple M4 GPU. Every variant shares the U-Net backbone, auxiliary class head, training budget, and data. Checkpoints use validation patient Dice, with a fixed 0.5 threshold. The 20-epoch validation pilot prompted the longer budget before any test scoring.

## Measured test results

| Model | Parameters | Patient Dice | Patient IoU | Boundary F1 |
|---|---:|---:|---:|---:|
| Compact U-Net | 488,100 | 59.79% | 49.82% | 60.71% |
| SoftROI ablation | 488,229 | 61.97% | 52.04% | 62.73% |
| SRB-U-Net | 492,167 | 63.17% | 53.76% | 65.79% |

Values are arithmetic averages of the two independently trained seeds. Patient Dice/IoU first average slices within each patient. Boundary F1 uses a two-pixel tolerance at 128 x 128 and averages slices.

SRB versus baseline: **+3.38 percentage points in patient Dice**. Paired patient bootstrap 95% interval: **[-0.18, +6.74] points**. 13/15 patients improved after averaging the two seeds. This interval reflects the held-out patients, not all uncertainty from training or choosing a different split.

The full proposal has 492,167 parameters, versus 488,100 in the baseline, an increase of 0.83%.

## Suggested 60-second update

“I identified a limitation in the base paper's detection-then-crop pipeline: a bounding box can restrict the context available to the segmenter. I implemented SoftROI-Boundary U-Net, which predicts a soft tumor region inside one network and uses it to guide skip features while retaining the full image. A boundary branch helps a residual head refine pixels near uncertain predictions.

I tested the proposal against a compact U-Net and a region-only ablation using a patient-separated subset. Across two seeds, baseline patient Dice was 59.79% and the full model reached 63.17%. The region-only ablation reached 61.97%. These are preliminary measurements. Next I will test more patient splits and separate the effects of boundary supervision and refinement.”

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
