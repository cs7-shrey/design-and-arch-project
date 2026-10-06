# SoftROI-Boundary U-Net: preliminary experiment

## Research question

Can a small U-Net preserve whole-image context while using a learned coarse tumor region and boundary-guided residual refinement to improve segmentation on limited Figshare data?

This is a proposed project architecture. Attention gates, coarse supervision, boundary supervision, and uncertainty proxies have prior art. We do not claim first-ever use, publication-level novelty, or state-of-the-art performance.

## Architectural hypothesis

The base paper detects a bounding box with YOLOv5, crops a region, and segments it with U-Net. It acknowledges that selecting the highest-confidence box may lose parts of a tumor. Our alternative retains the complete MRI and replaces the crop with a soft feature gate. This pilot tests segmentation and auxiliary tumor-type classification; it does not reproduce YOLOv5 detection or its mAP.

Shared backbone: one-channel input, 128 x 128 resolution, encoder widths 16/32/64/128, three decoder levels, bilinear upsampling, double convolution blocks with GroupNorm and SiLU. All variants have the same segmentation and three-class classification heads. No pretrained weights.

1. **Baseline:** compact U-Net plus an auxiliary tumor-type classification head.
2. **SoftROI:** same network, plus a 16 x 16 coarse tumor head. Upsampled coarse probability R modulates each encoder skip as S' = S * (0.5 + R). The nonzero floor retains contextual features.
3. **SRB-U-Net:** SoftROI plus a boundary head and a residual refinement head. Given provisional tumor probability p, define u = 4p(1-p). The refiner receives final decoder features, R, predicted boundary probability, and u. Final logit = provisional logit + u * learned residual. u is a heuristic uncertainty proxy, not calibrated epistemic uncertainty.

Only MRI intensities enter inference. Ground-truth masks supervise the segmentation, coarse ROI, and morphological boundary objectives during training.

## Frozen data construction

- Source: supplied MATLAB Figshare files and cvind.mat.
- Split RNG seed: 2026. Original folds 1/2/3 supply training patients, fold 4 validation, fold 5 test.
- Randomly choose 16/4/5 patients per class for train/validation/test and at most 10 randomly selected slices per patient.
- Result: train 48 patients / 398 slices; validation 12 / 117; test 15 / 129.
- Patient classes are balanced by selected patient count. Slice counts are not exactly balanced.
- All MATLAB image and mask arrays are transposed consistently on HDF5 loading.
- Normalize each image by the 99th percentile of its positive intensities and clip to [0,1]. Resize MRI with bilinear interpolation and mask with nearest neighbor.
- No skull stripping. No augmentation before splitting. Training-only flips, right-angle rotations, and intensity scaling.
- The subset manifest records every file and patient, and its SHA-256 is stored with each run.

## Comparison controls

- AdamW, learning rate 0.001, weight decay 0.0001, cosine decay to 0.0001, batch size 16.
- Fixed seeds 17 and 29, common backbone initialization copied across variants, identical ordering and augmentation stream per seed.
- Same epoch budget per model within a comparison. Initial budget: 20 epochs. Any larger validation-motivated budget must be recorded separately before test evaluation.
- Core loss = 0.5 weighted BCE (positive weight 3) + 0.5 soft Dice + 0.15 classification CE.
- SoftROI adds 0.25 coarse segmentation loss, using max-pooled masks to preserve small tumor regions.
- SRB adds 0.10 boundary loss (same BCE/Dice form, positive weight 8). Boundaries are 3 x 3 dilation minus erosion.
- Save the checkpoint with highest validation patient-mean Dice. Fixed probability threshold 0.5 for every model; no threshold search or connected-component cleanup.
- The ablation isolates the complete boundary/refinement module and its auxiliary objective. It does not separate boundary loss from residual architecture effects.

## Evaluation plan

- Primary metric: mean patient Dice (average slices within each patient, then average patients).
- Additional metrics: slice Dice, IoU, pixel precision/recall, boundary F1 at 2-pixel tolerance on the 128 x 128 grid, class accuracy and macro F1.
- Report all planned variants and seeds, including regressions. Seed-averaged model scores are averages of separate runs, not prediction ensembles.
- Use paired patient bootstrap resampling for the SRB-minus-baseline Dice interval, averaging seeds within each patient before resampling. Two seeds do not characterize training variance fully.
- Report per-class segmentation and small-tumor segmentation, using the training-set median mask area as the small-tumor cutoff.
- Keep paper metrics separate: its 88.1% Dice and 89.5% mAP use a different split, preprocessing, architecture, and budget. Our values are not a direct benchmark comparison.

## Prior work

- Ahsan et al., *Brain tumor detection and segmentation using deep learning*, published online 2024, journal issue 2025. https://doi.org/10.1007/s10334-024-01203-5 (local source: papers/brain-tumor-base.pdf)
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, 2018. https://arxiv.org/abs/1804.03999
- Xu et al., *Boundary guidance network for medical image segmentation*, 2024. https://www.nature.com/articles/s41598-024-67554-0
- Li et al., *Category Guided Attention Network for Brain Tumor Segmentation in MRI*, 2022. https://arxiv.org/abs/2203.15383

The search establishes relevant precedent and is not an exhaustive novelty review.

## Budget decision before held-out evaluation

The 20-epoch validation pilot reached its best scores at epochs 19-20 for the first completed models (baseline seed 17: 0.5404, baseline seed 29: 0.5369, SoftROI seed 17: 0.6125, SRB seed 17: 0.6118). To allow more learning, the final experiment uses **60 epochs for every variant and both seeds**, starting from the same shared initializations. The initial runs remain under `runs/pilot`, and the final runs under `runs/main`. The test set has not been scored at this decision point. No architecture, loss weight, threshold, or data subset changes accompany the longer budget. Main-run checkpoints still use validation patient Dice for selection.
