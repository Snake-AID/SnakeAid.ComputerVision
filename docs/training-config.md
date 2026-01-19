# Training configuration

## Core settings

```python
epochs = 120
batch = 32
imgsz = 640
```

- An epoch is one full pass over the dataset. Training can stop earlier via early stopping.
- Batch size balances stability vs GPU memory.
- Images are resized to a fixed resolution for batching and compute efficiency.

## Optimization

```python
optimizer = "AdamW"
lr0 = 0.002
lrf = 0.01
cos_lr = True
warmup_epochs = 3
patience = 15
```

- AdamW converges quickly and handles noisy gradients.
- Cosine decay starts aggressively and cools down near the end; warmup stabilizes the first few epochs.
- Early stopping halts training if validation does not improve for 15 epochs.

## Data augmentation

```python
hsv_h, hsv_s, hsv_v
degrees, translate, scale, shear
mosaic, mixup, copy_paste
```

Augmentations improve robustness and reduce overfitting by exposing the model to varied imagery.

### Mosaic

```python
mosaic = 1.0
close_mosaic = 10
```

- Combines four images to increase object density and small-object performance.
- Disabled in the final epochs to fine-tune on natural images.

### MixUp and Copy-Paste

```python
mixup = 0.10
copy_paste = 0.10
```

Gentle blending to reduce overfitting and simulate complex scenes without confusing the model.

## Model scale

From training logs:

```
231 layers
20,069,970 parameters
68.3 GFLOPs
```

- Parameter count shows capacity; GFLOPs indicate inference cost and expected runtime needs.

## Evaluation metrics

Validation reports:

```
Box(P)   R   mAP50   mAP50-95
```

- Precision (P): fraction of predicted boxes that are correct.
- Recall (R): fraction of real objects that were detected.
- mAP50: performance at IoU 0.5; mAP50-95 averages stricter thresholds for a harder score.
