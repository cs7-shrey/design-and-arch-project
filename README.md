# Brain tumor segmentation: SoftROI-Boundary U-Net

An implemented architecture proposal and reproducible small-subset experiment based on the supplied Figshare MRI dataset. The project compares a compact U-Net, a soft-region ablation, and a network that adds boundary-conditioned residual refinement. **This is preliminary research, not an established novel method or a clinical tool.**

## Start here for the presentation

- `output/Brain_Tumor_Progress_Update.pptx`: editable slides and speaker notes.
- `output/Brain_Tumor_Progress_Update.pdf`: portable visual backup of the slides.
- `output/PRESENTATION_BRIEF.md`: actual results, a 60-second script, likely questions, and limitations.
- `output/figures/`: learning curves, predictions, model maps, and a failure case.
- `Project_Update.ipynb`: walkthrough of the saved results and an MRI-only demo.
- `ARCHITECTURE.md`: model diagram and explanation of each module.
- `output/architecture/`: presentation-ready architecture diagrams in PNG/SVG and editable PowerPoint format.
- `EXPERIMENT_PROTOCOL.md`: architecture, comparison controls, metric definitions, and prior work.
- `runs/main/results.json`: all six runs, aggregate metrics, per-class results, and a paired patient bootstrap interval.

## Architecture

Input: one-channel 128 x 128 MRI. Four encoder scales use 16, 32, 64, and 128 channels. A shared U-Net decoder produces a provisional segmentation. Every comparison also has a global-average-pooled three-class auxiliary classification head.

**Baseline (488,100 parameters):** compact U-Net with GroupNorm and SiLU.

**SoftROI (488,229):** a supervised coarse tumor map R from the deepest encoder reweights each skip tensor: `guided_skip = skip * (0.5 + resized_R)`. All spatial locations remain available.

**SRB-U-Net (492,167):** adds a boundary head and a residual refiner. The refiner consumes decoder features, predicted ROI probability, predicted boundary probability, and `u = 4*p*(1-p)`. Its correction is weighted by u and added to the provisional logit. u is a heuristic confidence proxy, not a calibrated uncertainty estimate. The residual output layer starts at zero.

Ground-truth masks are used for losses and evaluation only. They never enter inference. A predicted mask can produce a bounding box for visualization, but **we do not measure object-detector mAP**.

## Data

Original data: 3,064 slices, 233 patients, three tumor classes. MATLAB labels 1/2/3 map to Python classes 0/1/2 (meningioma/glioma/pituitary).

The frozen subset uses 48/12/15 patients and 398/117/129 slices for training/validation/test. Original folds 1-3/4/5 define the respective patient pools. The subset selects the same number of patients per class, with a maximum of ten slices per patient. Images use per-image 99th percentile intensity normalization and bilinear resizing. Masks use nearest-neighbor resizing. No skull stripping.

The original data contain no healthy class. This experiment concerns segmenting and classifying tumors in tumor-positive slices.

## Setup and reproduce

The existing `.venv` has all dependencies installed. For a new environment:

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Run from the project root:

```bash
.venv/bin/python -m src.prepare
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u -m src.train --epochs 60 --output runs/reproduction
.venv/bin/python -m src.evaluate --run runs/reproduction
.venv/bin/python -m src.report --run runs/reproduction
```

Training automatically uses Apple MPS when available, otherwise CPU. Seeds fix initialization, sample order, and augmentation streams, but exact bitwise reproducibility across hardware/library versions is not guaranteed. Completed identical runs are skipped. Use a new output folder when changing configuration. `src.report` regenerates presentation figures and the brief; the existing PPTX remains unchanged until rebuilt separately.

All models have equal training budgets and share common initial weights per seed. The selected checkpoint maximizes **validation patient-mean Dice**, with a fixed 0.5 probability threshold. Test evaluation requires every planned model run to finish. Training never evaluates the test set. The initial 20-epoch validation pilot is retained under `runs/pilot`; it motivated the final 60-epoch budget before test scoring.

## Run saved-model inference

```bash
.venv/bin/python -m src.predict \
  --checkpoint runs/main/srb_seed17/checkpoints/best.pt \
  --input data/mat/1.mat \
  --output output/demo
```

This produces a mask, overlay, probability array, and JSON class scores / mask-derived bounding box. Any image may be demonstrated, but **only files listed in the manifest's test split count as held-out evidence**. The delivered demo identifies its source file explicitly.

## Interpretation

The primary outcome averages Dice within each patient and then across patients. Additional outcomes include IoU, boundary F1 within two pixels at the evaluation resolution, pixel precision/recall, class accuracy, and macro F1. Reported aggregate scores average separate seeded runs, not ensemble predictions. Patient bootstrap intervals do not capture all uncertainty from training and dataset splitting.

The ablation adds architecture and auxiliary supervision together. A future study should isolate these effects separately. Fifteen test patients, two seeds, and one split do not establish robust superiority. Paper scores are not directly comparable because the dataset split, resolution, preprocessing, models, and training budget differ.

## Files

| Path | Purpose |
|---|---|
| `src/models.py` | Three variants and loss functions |
| `src/prepare.py` | Deterministic patient-separated subset |
| `src/train.py` | Training and validation checkpoint selection |
| `src/evaluate.py` | Held-out metrics and paired bootstrap |
| `src/predict.py` | MRI-only inference |
| `src/report.py` | Figures and result-driven presentation brief |
| `tests/test_core.py` | Metrics, gradients, boundary targets, split checks |
| `data/processed/manifest.json` | Sample-level provenance and split membership |
| `runs/main/*/history.csv` | Per-epoch logs |
| `runs/main/*/checkpoints/best.pt` | Selected trained models |
| `runs/main/test_per_slice.csv` | Predictions scored per slice |
| `.presentation-build/build.mjs` | PPTX generation using bundled Artifact Tool |

Source paper and research precedents are linked in `EXPERIMENT_PROTOCOL.md` and in the slides' notes.

The original main-run results can be independently checked with `.venv/bin/python scripts/verify_results.py`, which recomputes Dice from saved probabilities and checks the validation-selected epoch for all six runs.
