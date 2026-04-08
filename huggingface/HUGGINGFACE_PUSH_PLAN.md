# Push SnakeAI Models And Datasets To Hugging Face

## Summary

- Use Hugging Face account `the-khiem7`.
- Create public, separate repos so every dataset version and every `.pt` checkpoint has its own card.
- Use `huggingface_hub` Python API as the primary implementation path because this workflow needs repo creation, generated cards, model-dataset linking, exclude rules, dry-run manifest, and post-upload verification. Use `hf` CLI only as an optional fallback/manual verification tool after it is installed and authenticated.
- Require local auth via `HF_TOKEN` or `huggingface_hub.login(...)`.
- Do not deep-scan images/labels; use only folder names, counts, `data.yaml`, README metadata, file size, and model hashes.

## Current Status

- Completed upload and verification on 2026-04-08 with `rtk python scripts/push_snakeaid_to_hf.py --root . --upload --verify`.
- Created/reused 5 public dataset repos and 6 public model repos under `the-khiem7`.
- All dataset repos have `README.md` cards and no forbidden `.git`, `.cache`, or `.zip` files.
- All model repos have `README.md` cards and exactly their intended `.pt` checkpoint.
- `SnakeAI_5k_120e.pt` was intentionally ignored and not uploaded.
- Final verification output reported `pt_ok: true` for all 6 model repos.
- The upload script remains optimized for retries:
  - Default dataset upload mode is `batched`, uploading `train/valid/test` and `images/labels` folders separately.
  - Complete remote dataset/model repos are skipped by default based on Hub file state.
  - `--force` can re-upload even when a repo appears complete.
  - `--dataset-upload-mode large-folder` remains available as an explicit fallback, with optional `--large-folder-workers`.
- Card generator was improved after initial review of the Hub pages:
  - Dataset cards now include key details, split table, file layout, loading example, class table, provenance, related models, limitations, and safety notes.
  - Model cards now include model summary, safety note, checkpoint metadata table, intended/out-of-scope use, dataset linkage, download/Ultralytics usage snippets, evaluation notes, and reproducibility caveats.
  - Added `--cards-only` to update only generated `README.md` cards without re-uploading images, labels, or `.pt` files.
  - Moved operational scripts into `scripts/`.
  - Added `scripts/refine_hf_cards.py` as a dedicated README card refinement entrypoint.
  - Local validation passed for all generated dataset/model card YAML metadata.
  - Card-only upload was not run from this assistant shell because `HF_TOKEN` was not available in the tool environment; run the command below from a terminal with `HF_TOKEN` set.
- Source-of-truth verification was completed before local cleanup:
  - Generated report: `HUGGINGFACE_SOURCE_OF_TRUTH_REPORT.json`.
  - `all_ok: true`.
  - All 5 dataset repos match the expected uploadable payload files from local folders, excluding local upload cache/control files such as `.cache`, `.git`, `.zip`, and `.gitattributes`.
  - All 6 model repos contain exactly the expected `.pt`; Hugging Face LFS size and SHA256 match the local manifest.
  - No forbidden dataset files (`.git`, `.cache`, `.zip`) were found on Hub.

Card-only update command:

```powershell
rtk python scripts/refine_hf_cards.py --root . --verify
```

## Roadmap

- [x] Define Hugging Face repo layout: public, separate repos for each dataset version and model checkpoint.
- [x] Finalize model brand/subbrand: `SnakeAid Detect`.
- [x] Finalize model naming convention: version tokens before numbers (`v4`, `v6`, `h2`), epoch/patience tokens after numbers (`10e`, `40e`, `120e`, `10p`), dataset token at the end.
- [x] Decide to ignore `SnakeAI_5k_120e.pt`.
- [x] Confirm local Hugging Face auth with `huggingface_hub` and verify account is `the-khiem7`.
- [x] Build an idempotent upload script using `huggingface_hub`.
- [x] Generate a dry-run manifest with dataset counts, model hashes, repo ids, and model-dataset links.
- [x] Generate dataset cards and model cards.
- [x] Create or reuse the 5 dataset repos and 6 model repos.
- [x] Upload datasets with `.git/**`, `*.cache`, `*.zip`, and staging artifacts excluded.
- [x] Upload the 6 selected `.pt` model checkpoints.
- [x] Verify all 11 repos, cards, uploaded files, excludes, and model-dataset links after upload.
- [x] Record final Hugging Face URLs and upload summary in this document.

## Hugging Face Repos To Create

Datasets:

- `the-khiem7/snakeaid-yolov12-300-masking` <- `SnakeAid-YOLOv12-300Masking`
- `the-khiem7/snakeaid-yolov12-5000-bbox` <- `SnakeAid-YOLOv12-5000bbox`
- `the-khiem7/snakeaid-yolov12-5000-masking` <- `SnakeAid-YOLOv12-5000Masking`
- `the-khiem7/snakeaid-yolov12-5291-bbox` <- `SnakeAid-YOLOv12-5291BBox`
- `the-khiem7/snakeaid-yolov12-5291-bbox-complete` <- `SnakeAid-YOLOv12-5291Bbox-Complete`

Models:

- `the-khiem7/snakeaid-detect-yolov12n-10e-300masking` <- `SnakeAid_SnakeDetector_YOLOv12n_300Masking_10epoch_Khiem.pt`
- `the-khiem7/snakeaid-detect-yolov12n-10e-5000masking` <- `SnakeAid_SnakeDetector_YOLOv12n_5000Masking_10epoch_Khiem.pt`
- `the-khiem7/snakeaid-detect-yolov12s-17e-5000bbox` <- `SnakeAid_SnakeDetector_YOLOv12s_5000Bbox_17epoch_Nhan.pt`
- `the-khiem7/snakeaid-detect-yolov12-v4-5000bbox` <- `SnakeTraining_V4_YOLOv12_Khiem_Bbox5000_20251213_1828.pt`
- `the-khiem7/snakeaid-detect-yolov12-v6-40e-10p-5291bbox-complete` <- `SnakeTraining_V6_YOLOv12_Khiem_Bbox5291Complete_40epoch_10patience_20251215_0303.pt`
- `the-khiem7/snakeaid-detect-yolov12-h2-120e-10p-5291bbox-complete` <- `SnakeTraining_H2_YOLOv12_Hao_Bbox5291Complete_120epoch_10patience_20251215_1346.pt`

## Final Hugging Face URLs

Datasets:

- https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-300-masking - verified `file_count: 711`
- https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5000-bbox - verified `file_count: 10587`
- https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5000-masking - verified `file_count: 9919`
- https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5291-bbox - verified `file_count: 10587`
- https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5291-bbox-complete - verified `file_count: 10585`

Models:

- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12n-10e-300masking - verified `.pt`: `SnakeAid_SnakeDetector_YOLOv12n_300Masking_10epoch_Khiem.pt`
- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12n-10e-5000masking - verified `.pt`: `SnakeAid_SnakeDetector_YOLOv12n_5000Masking_10epoch_Khiem.pt`
- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12s-17e-5000bbox - verified `.pt`: `SnakeAid_SnakeDetector_YOLOv12s_5000Bbox_17epoch_Nhan.pt`
- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-v4-5000bbox - verified `.pt`: `SnakeTraining_V4_YOLOv12_Khiem_Bbox5000_20251213_1828.pt`
- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-v6-40e-10p-5291bbox-complete - verified `.pt`: `SnakeTraining_V6_YOLOv12_Khiem_Bbox5291Complete_40epoch_10patience_20251215_0303.pt`
- https://huggingface.co/the-khiem7/snakeaid-detect-yolov12-h2-120e-10p-5291bbox-complete - verified `.pt`: `SnakeTraining_H2_YOLOv12_Hao_Bbox5291Complete_120epoch_10patience_20251215_1346.pt`

## Upload And Linking Behavior

- Generate a `README.md` dataset card for each dataset repo with YOLOv12 format, train/valid/test counts, class names from `data.yaml`, Roboflow source/license, and related models.
- Generate a `README.md` model card for each model repo with checkpoint metadata, hash/size, intended use, linked dataset, and related dataset metadata.
- Upload dataset folders as raw YOLO structure in batched split/kind commits by default and exclude `.git/**`, `*.cache`, `*.zip`, and local staging artifacts.
- Upload each `.pt` file to its own model repo.
- Map model to dataset by filename tokens: `300Masking`, `5000Masking`, `5000Bbox`/`Bbox5000`, and `Bbox5291Complete`.
- Ignore `SnakeAI_5k_120e.pt`; do not upload it and do not create a Hugging Face model repo for it.

## Test Plan

- Confirm token works with `HfApi().whoami()` and abort if account is not `the-khiem7`.
- Print a dry manifest before upload: repos, local paths, file counts, model sizes, hashes, and links.
- Verify all 11 repos exist after upload.
- Verify each repo has `README.md`.
- Verify model repos contain exactly their intended `.pt`.
- Verify dataset repos do not contain `.cache`, `.git`, or `.zip`.
- Verify model cards link to dataset repos and dataset cards link back to related model repos.

## Assumptions

- Repos are public.
- Separate repos are required so each version has a proper Hub card.
- Dataset content should not be deeply inspected.
- Roboflow `Public Domain` license metadata can be reflected in dataset cards.
- Existing Hub search found no SnakeAI/SnakeAid repos under `the-khiem7`, so implementation should create repos if missing and update if they appear later.
