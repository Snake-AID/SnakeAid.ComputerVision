# SnakeAid Hugging Face Migration Record

This repository is the operational record for migrating the SnakeAid/SnakeAI local datasets and PyTorch checkpoints to Hugging Face Hub. It documents what was uploaded, which repositories now exist on Hugging Face, which local files were intentionally excluded, which scripts were created, and what has been verified before treating Hugging Face as the source of truth.

This README is not a model card or a dataset card. The actual Hugging Face cards live inside the corresponding Hub repositories. This file is the local project history and maintenance guide.

## Table Of Contents

- [Current Status](#current-status)
- [Source Of Truth Statement](#source-of-truth-statement)
- [Hugging Face Repositories](#hugging-face-repositories)
- [Local Workspace Artifacts](#local-workspace-artifacts)
- [Operational Scripts](#operational-scripts)
- [Migration Timeline](#migration-timeline)
- [Known Issues And Fixes](#known-issues-and-fixes)
- [Verification Evidence](#verification-evidence)
- [Recommended Next Workflow](#recommended-next-workflow)
- [Before Deleting Local Files](#before-deleting-local-files)

## Current Status

- Hugging Face account: `the-khiem7`.
- Main upload and verification date: `2026-04-08`.
- Uploaded public dataset repositories: `5`.
- Uploaded public model repositories: `6`.
- Intentionally excluded model file: `SnakeAI_5k_120e.pt`.
- Source-of-truth verification report: `HUGGINGFACE_SOURCE_OF_TRUTH_REPORT.json`.
- Final source-of-truth result: `all_ok: true`.

All selected dataset and model payloads have been uploaded to Hugging Face and verified against the local manifest. The local workspace is now primarily useful for operational history, scripts, manifest/report files, and any local-only artifacts that were intentionally not uploaded.

## Source Of Truth Statement

Hugging Face is now a valid source of truth for the selected SnakeAid datasets and model checkpoints listed in this document.

That statement has a precise scope:

- It covers the 5 dataset repositories listed below.
- It covers the 6 model repositories listed below.
- It covers dataset payload files after excluding local control/cache files such as `.cache`, `.git`, `.zip`, and `.gitattributes`.
- It covers the uploaded `.pt` files by size and LFS SHA256 match.
- It does not cover local workflow files such as `scripts/`, `HUGGINGFACE_PUSH_PLAN.md`, `HUGGINGFACE_PUSH_MANIFEST.json`, `HUGGINGFACE_SOURCE_OF_TRUTH_REPORT.json`, `skills-lock.json`, or `.agents/skills/`.
- It does not cover `SnakeAI_5k_120e.pt`, because that checkpoint was intentionally ignored and not uploaded.

## Hugging Face Repositories

### Dataset Repositories

| Hugging Face repo | Original local folder | Verification state |
| --- | --- | --- |
| https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-300-masking | `SnakeAid-YOLOv12-300Masking` | Verified |
| https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5000-bbox | `SnakeAid-YOLOv12-5000bbox` | Verified |
| https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5000-masking | `SnakeAid-YOLOv12-5000Masking` | Verified |
| https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5291-bbox | `SnakeAid-YOLOv12-5291BBox` | Verified |
| https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5291-bbox-complete | `SnakeAid-YOLOv12-5291Bbox-Complete` | Verified |

### Model Repositories

| Hugging Face repo | Uploaded checkpoint | Verification state |
| --- | --- | --- |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12n-10e-300masking | `SnakeAid_SnakeDetector_YOLOv12n_300Masking_10epoch_Khiem.pt` | Verified |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12n-10e-5000masking | `SnakeAid_SnakeDetector_YOLOv12n_5000Masking_10epoch_Khiem.pt` | Verified |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12s-17e-5000bbox | `SnakeAid_SnakeDetector_YOLOv12s_5000Bbox_17epoch_Nhan.pt` | Verified |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-v4-5000bbox | `SnakeTraining_V4_YOLOv12_Khiem_Bbox5000_20251213_1828.pt` | Verified |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-v6-40e-10p-5291bbox-complete | `SnakeTraining_V6_YOLOv12_Khiem_Bbox5291Complete_40epoch_10patience_20251215_0303.pt` | Verified |
| https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-h2-120e-10p-5291bbox-complete | `SnakeTraining_H2_YOLOv12_Hao_Bbox5291Complete_120epoch_10patience_20251215_1346.pt` | Verified |

### Authentication Requirement

Any command that mutates Hugging Face Hub needs local Hugging Face authentication. The expected account is `the-khiem7`.

Typical PowerShell setup:

```powershell
$env:HF_TOKEN="hf_..."
rtk python scripts\push_snakeaid_to_hf.py --root . --upload --verify
```

The scripts are expected to abort if the authenticated account does not match `the-khiem7`.

## Migration Timeline

1. The initial goal was to push multiple local dataset folders and multiple local `.pt` trained model checkpoints to Hugging Face.
2. The repository strategy was standardized: one public Hugging Face repository per dataset version and one public Hugging Face repository per selected checkpoint.
3. The implementation chose `huggingface_hub` Python APIs because the workflow needed generated cards, repo creation/reuse, manifest generation, linking, exclude rules, and verification.
4. A dry-run manifest was generated at `HUGGINGFACE_PUSH_MANIFEST.json`.
5. The first upload attempt failed when Hugging Face rejected the generated card metadata value `license: public-domain`.
6. The license metadata was corrected to the Hugging Face-supported identifier `cc0-1.0`.
7. Uploading large dataset folders in one pass produced warnings and could appear stuck. The script was updated to use smaller batched uploads and skip already-complete Hub repositories.
8. Upload and verification completed for 5 dataset repositories and 6 model repositories.
9. The generated model and dataset cards were later improved with richer content and more explicit model-dataset linking.
10. Card refinement was split into `scripts/refine_hf_cards.py` so README card updates can be done without touching dataset/model payloads.
11. A source-of-truth verification pass was run before any local deletion decision.
12. `HUGGINGFACE_SOURCE_OF_TRUTH_REPORT.json` was generated with `all_ok: true`.

## Known Issues And Fixes

### Invalid Hugging Face License Metadata

Problem:

```yaml
license: public-domain
```

Hugging Face card metadata validation rejected this value.

Fix:

```yaml
license: cc0-1.0
```

### Large Folder Uploads Could Look Stuck

The Hugging Face client warned that uploading a large folder at once could take time or fail. During dataset upload, progress could show long periods with little visible movement.

Fixes applied:

- Dataset uploads were changed to smaller batches.
- Completed Hub repositories are skipped by default.
- `--force` remains available if an explicit re-upload is needed.

### Card-Only Update Encountered A 504

During one card-only refinement attempt, Hugging Face returned:

```text
HTTP Error 504 thrown while requesting POST https://huggingface.co/api/models/the-khiem7/snakeaid-detect-yolov12-v4-5000bbox/preupload/main
Retrying in 1s [Retry 1/5].
```

This was an API-side transient failure during README upload, not evidence that the model payload was missing. If card editing remains unstable through the API path, the recommended fallback is to clone the relevant Hugging Face repository locally, edit `README.md`, and push through Git/Git LFS.

## Verification Evidence

The final source-of-truth report confirmed:

- Every selected dataset repo has a `README.md`.
- Every selected model repo has a `README.md`.
- Dataset payload files on Hub match the expected local uploadable files.
- Dataset verification excludes local control/cache files such as `.cache`, `.git`, `.zip`, and `.gitattributes`.
- No forbidden dataset files `.git`, `.cache`, or `.zip` were found on Hub.
- Each model repository contains exactly the expected `.pt` checkpoint.
- Hugging Face LFS size and SHA256 match the local manifest for all 6 uploaded checkpoints.

Summary by dataset:

| Dataset repo | Expected payload files | Remote payload files | Remote total files | Image count | Text/label count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `the-khiem7/snakeaid-yolov12-300-masking` | 710 | 710 | 711 | 353 | 355 |
| `the-khiem7/snakeaid-yolov12-5000-bbox` | 10586 | 10586 | 10587 | 5291 | 5293 |
| `the-khiem7/snakeaid-yolov12-5000-masking` | 9918 | 9918 | 9919 | 4957 | 4959 |
| `the-khiem7/snakeaid-yolov12-5291-bbox` | 10586 | 10586 | 10587 | 5291 | 5293 |
| `the-khiem7/snakeaid-yolov12-5291-bbox-complete` | 10584 | 10584 | 10585 | 5291 | 5291 |

Summary by model:

| Model repo | Checkpoint state |
| --- | --- |
| `the-khiem7/snakeaid-detect-yolov12n-10e-300masking` | `.pt` present, size matched, SHA256 matched |
| `the-khiem7/snakeaid-detect-yolov12n-10e-5000masking` | `.pt` present, size matched, SHA256 matched |
| `the-khiem7/snakeaid-detect-yolov12s-17e-5000bbox` | `.pt` present, size matched, SHA256 matched |
| `the-khiem7/snakeaid-detect-yolov12-v4-5000bbox` | `.pt` present, size matched, SHA256 matched |
| `the-khiem7/snakeaid-detect-yolov12-v6-40e-10p-5291bbox-complete` | `.pt` present, size matched, SHA256 matched |
| `the-khiem7/snakeaid-detect-yolov12-h2-120e-10p-5291bbox-complete` | `.pt` present, size matched, SHA256 matched |