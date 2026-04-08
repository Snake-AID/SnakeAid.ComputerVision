from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


ACCOUNT = "the-khiem7"
BASE_URL = "https://huggingface.co"
DATASET_URL = "https://huggingface.co/datasets"

EXCLUDE_PATTERNS = [
    ".git/*",
    ".git/**",
    "*.cache",
    "*.zip",
    "__pycache__/*",
    "__pycache__/**",
    "HUGGINGFACE_PUSH_MANIFEST.json",
]


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    local_dir: str
    title: str


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    title: str
    dataset_repo_id: str
    architecture: str
    epochs: str | None = None
    patience: str | None = None
    trained_by: str | None = None
    version: str | None = None


DATASETS = [
    DatasetSpec(
        repo_id=f"{ACCOUNT}/snakeaid-yolov12-300-masking",
        local_dir="SnakeAid-YOLOv12-300Masking",
        title="SnakeAid YOLOv12 300 Masking",
    ),
    DatasetSpec(
        repo_id=f"{ACCOUNT}/snakeaid-yolov12-5000-bbox",
        local_dir="SnakeAid-YOLOv12-5000bbox",
        title="SnakeAid YOLOv12 5000 BBox",
    ),
    DatasetSpec(
        repo_id=f"{ACCOUNT}/snakeaid-yolov12-5000-masking",
        local_dir="SnakeAid-YOLOv12-5000Masking",
        title="SnakeAid YOLOv12 5000 Masking",
    ),
    DatasetSpec(
        repo_id=f"{ACCOUNT}/snakeaid-yolov12-5291-bbox",
        local_dir="SnakeAid-YOLOv12-5291BBox",
        title="SnakeAid YOLOv12 5291 BBox",
    ),
    DatasetSpec(
        repo_id=f"{ACCOUNT}/snakeaid-yolov12-5291-bbox-complete",
        local_dir="SnakeAid-YOLOv12-5291Bbox-Complete",
        title="SnakeAid YOLOv12 5291 BBox Complete",
    ),
]

MODELS = [
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12n-10e-300masking",
        filename="SnakeAid_SnakeDetector_YOLOv12n_300Masking_10epoch_Khiem.pt",
        title="SnakeAid Detect YOLOv12n 10e 300Masking",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-300-masking",
        architecture="YOLOv12n",
        epochs="10",
        trained_by="Khiem",
    ),
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12n-10e-5000masking",
        filename="SnakeAid_SnakeDetector_YOLOv12n_5000Masking_10epoch_Khiem.pt",
        title="SnakeAid Detect YOLOv12n 10e 5000Masking",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-5000-masking",
        architecture="YOLOv12n",
        epochs="10",
        trained_by="Khiem",
    ),
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12s-17e-5000bbox",
        filename="SnakeAid_SnakeDetector_YOLOv12s_5000Bbox_17epoch_Nhan.pt",
        title="SnakeAid Detect YOLOv12s 17e 5000BBox",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-5000-bbox",
        architecture="YOLOv12s",
        epochs="17",
        trained_by="Nhan",
    ),
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12-v4-5000bbox",
        filename="SnakeTraining_V4_YOLOv12_Khiem_Bbox5000_20251213_1828.pt",
        title="SnakeAid Detect YOLOv12 v4 5000BBox",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-5000-bbox",
        architecture="YOLOv12",
        trained_by="Khiem",
        version="v4",
    ),
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12-v6-40e-10p-5291bbox-complete",
        filename="SnakeTraining_V6_YOLOv12_Khiem_Bbox5291Complete_40epoch_10patience_20251215_0303.pt",
        title="SnakeAid Detect YOLOv12 v6 40e 10p 5291BBox Complete",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-5291-bbox-complete",
        architecture="YOLOv12",
        epochs="40",
        patience="10",
        trained_by="Khiem",
        version="v6",
    ),
    ModelSpec(
        repo_id=f"{ACCOUNT}/snakeaid-detect-yolov12-h2-120e-10p-5291bbox-complete",
        filename="SnakeTraining_H2_YOLOv12_Hao_Bbox5291Complete_120epoch_10patience_20251215_1346.pt",
        title="SnakeAid Detect YOLOv12 h2 120e 10p 5291BBox Complete",
        dataset_repo_id=f"{ACCOUNT}/snakeaid-yolov12-5291-bbox-complete",
        architecture="YOLOv12",
        epochs="120",
        patience="10",
        trained_by="Hao",
        version="h2",
    ),
]


def read_data_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {"classes": [], "roboflow": {}}
    in_roboflow = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            metadata["classes"] = ast.literal_eval(value)
            continue
        if line.startswith("nc:"):
            metadata["num_classes"] = int(line.split(":", 1)[1].strip())
            continue
        if line == "roboflow:":
            in_roboflow = True
            continue
        if in_roboflow and ":" in line:
            key, value = line.split(":", 1)
            metadata["roboflow"][key.strip()] = value.strip()
    return metadata


def count_dataset_files(dataset_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        counts[split] = {}
        for kind in ["images", "labels"]:
            folder = dataset_dir / split / kind
            counts[split][kind] = (
                sum(1 for path in folder.iterdir() if path.is_file())
                if folder.exists()
                else 0
            )
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_exclude(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDE_PATTERNS)


def count_uploadable_files(root: Path) -> int:
    count = 0
    for path in root.rglob("*"):
        if path.is_file() and not should_exclude(path.relative_to(root).as_posix()):
            count += 1
    return count


def dataset_manifest(root: Path, spec: DatasetSpec) -> dict[str, Any]:
    dataset_dir = root / spec.local_dir
    yaml_path = dataset_dir / "data.yaml"
    metadata = read_data_yaml(yaml_path)
    return {
        "repo_id": spec.repo_id,
        "type": "dataset",
        "title": spec.title,
        "local_dir": spec.local_dir,
        "url": f"{DATASET_URL}/{spec.repo_id}",
        "counts": count_dataset_files(dataset_dir),
        "uploadable_files": count_uploadable_files(dataset_dir),
        "num_classes": metadata.get("num_classes"),
        "classes": metadata.get("classes", []),
        "roboflow": metadata.get("roboflow", {}),
        "related_models": [model.repo_id for model in MODELS if model.dataset_repo_id == spec.repo_id],
    }


def model_manifest(root: Path, spec: ModelSpec) -> dict[str, Any]:
    model_path = root / spec.filename
    return {
        "repo_id": spec.repo_id,
        "type": "model",
        "title": spec.title,
        "filename": spec.filename,
        "url": f"{BASE_URL}/{spec.repo_id}",
        "dataset_repo_id": spec.dataset_repo_id,
        "dataset_url": f"{DATASET_URL}/{spec.dataset_repo_id}",
        "size_bytes": model_path.stat().st_size,
        "size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
        "sha256": sha256_file(model_path),
        "architecture": spec.architecture,
        "epochs": spec.epochs,
        "patience": spec.patience,
        "trained_by": spec.trained_by,
        "version": spec.version,
    }


def build_manifest(root: Path) -> dict[str, Any]:
    return {
        "account": ACCOUNT,
        "visibility": "public",
        "ignored_models": ["SnakeAI_5k_120e.pt"],
        "exclude_patterns": EXCLUDE_PATTERNS,
        "datasets": [dataset_manifest(root, spec) for spec in DATASETS],
        "models": [model_manifest(root, spec) for spec in MODELS],
    }


def dataset_card(item: dict[str, Any]) -> str:
    rows = [
        f"| {split} | {counts['images']} | {counts['labels']} |"
        for split, counts in item["counts"].items()
    ]
    class_rows = "\n".join(
        f"| {index} | `{name}` |" for index, name in enumerate(item["classes"])
    )
    related_models = "\n".join(
        f"- [{repo_id}]({BASE_URL}/{repo_id})" for repo_id in item["related_models"]
    )
    roboflow = item["roboflow"]
    roboflow_lines = "\n".join(
        f"- {key}: {value}" for key, value in roboflow.items()
    ) or "- Not provided in `data.yaml` for this local export."
    dataset_id = item["repo_id"].split("/", 1)[1]
    total_images = sum(counts["images"] for counts in item["counts"].values())
    total_labels = sum(counts["labels"] for counts in item["counts"].values())
    split_names = ", ".join(item["counts"].keys())

    return f"""---
license: cc0-1.0
task_categories:
- object-detection
tags:
- yolo
- yolov12
- snake-detection
- roboflow
pretty_name: {item["title"]}
---

# {item["title"]}

## Dataset Summary

This repository contains a YOLO-format SnakeAid object-detection dataset for snake detection experiments. It is organized as image/label pairs across `{split_names}` splits and is intended for training or evaluating YOLO-family detectors, including the related SnakeAid Detect YOLOv12 checkpoints linked below.

> Safety note: snake detection can be safety-critical in real-world use. Treat model outputs trained on this data as assistive signals only; do not use them as the sole basis for handling, approaching, or identifying a snake.

## Key Details

| Field | Value |
| --- | --- |
| Format | YOLO object detection |
| Splits | `{split_names}` |
| Images | {total_images} |
| Label files | {total_labels} |
| Classes | {item["num_classes"]} |
| License metadata | `cc0-1.0` |
| Uploadable local files | {item["uploadable_files"]} |

## Splits

| Split | Images | Labels |
| --- | ---: | ---: |
{chr(10).join(rows)}

## File Layout

```text
data.yaml
train/images/*.jpg
train/labels/*.txt
valid/images/*.jpg
valid/labels/*.txt
test/images/*.jpg
test/labels/*.txt
```

Each image is paired with a YOLO `.txt` label file using the same stem. Class names and split paths are defined in `data.yaml`.

## Classes

| Class ID | Name |
| ---: | --- |
{class_rows}

## Loading Example

```python
from huggingface_hub import snapshot_download

dataset_dir = snapshot_download(
    repo_id="{item["repo_id"]}",
    repo_type="dataset",
)
print(dataset_dir)
```

For YOLO training, point your training command at the downloaded `data.yaml`.

## Provenance

{roboflow_lines}

## Related Models

{related_models or "- None"}

## Limitations

- The dataset is provided as a local YOLO export, not as a fully curated benchmark.
- Class balance, duplicate images, annotation quality, and real-world geographic coverage have not been independently audited in this upload workflow.
- Performance can vary significantly with lighting, camera angle, occlusion, species similarity, and image quality.
- Use additional validation before deploying a detector trained on this data in field or safety-sensitive settings.
"""


def model_card(item: dict[str, Any]) -> str:
    extra = [
        ("Architecture", item["architecture"]),
        ("Version", item["version"]),
        ("Epochs", item["epochs"]),
        ("Patience", item["patience"]),
        ("Trained by", item["trained_by"]),
    ]
    metadata_rows = "\n".join(f"| {key} | {value} |" for key, value in extra if value)
    metadata_bullets = "\n".join(f"- {key}: {value}" for key, value in extra if value)
    dataset_name = item["dataset_repo_id"].split("/", 1)[1]
    return f"""---
license: other
library_name: ultralytics
pipeline_tag: object-detection
tags:
- yolo
- yolov12
- snake-detection
datasets:
- {item["dataset_repo_id"]}
base_model: ultralytics/yolov12
model-index:
- name: {item["title"]}
  results: []
---

# {item["title"]}

## Model Summary

This repository contains a SnakeAid Detect YOLOv12 checkpoint for snake object detection. The checkpoint is published as a `.pt` file and linked to the dataset version used for the corresponding experiment.

> Safety note: this model is intended for research, prototyping, and application experiments. Do not use it as the sole authority for snake identification, emergency response, or animal handling decisions.

## Checkpoint

| Field | Value |
| --- | --- |
| File | `{item["filename"]}` |
| Size | {item["size_mb"]} MB |
| SHA256 | `{item["sha256"]}` |
| Dataset | [`{item["dataset_repo_id"]}`]({item["dataset_url"]}) |
{metadata_rows}

## Intended Use

- Snake detection experiments with YOLOv12-compatible tooling.
- Comparing SnakeAid dataset versions and checkpoint variants.
- Downstream evaluation, prototyping, and transfer-learning baselines.

## Out-of-Scope Use

- Safety-critical snake identification without human/expert review.
- Medical, veterinary, emergency, or wildlife handling decisions based only on model output.
- Deployment to image domains not represented by the linked dataset without additional validation.

## Dataset

This model is linked to [`{item["dataset_repo_id"]}`]({item["dataset_url"]}).

The Hub metadata also includes this dataset in the `datasets` field so Hugging Face can display the training-data relationship.

## Usage Example

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="{item["repo_id"]}",
    filename="{item["filename"]}",
)
print(checkpoint_path)
```

If your environment supports Ultralytics-compatible YOLOv12 checkpoints:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

checkpoint_path = hf_hub_download(
    repo_id="{item["repo_id"]}",
    filename="{item["filename"]}",
)
model = YOLO(checkpoint_path)
results = model.predict("path/to/image.jpg")
```

## Evaluation

No standardized metrics were uploaded with this checkpoint. Evaluate it on a held-out split or a domain-specific benchmark before comparing it with other detectors or deploying it.

## Training And Reproducibility Notes

Known metadata:

{metadata_bullets or "- Training details were not encoded in the filename."}

The original training code, hyperparameters, random seed, and evaluation logs were not included in this upload.

## Related Dataset Version

- [`{dataset_name}`]({item["dataset_url"]})
"""


def upload_readme(api: HfApi, repo_id: str, repo_type: str | None, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Add generated Hugging Face card",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def repo_file_names(
    api: HfApi,
    repo_id: str,
    repo_type: str | None,
) -> set[str] | None:
    try:
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, files_metadata=False)
        else:
            info = api.model_info(repo_id, files_metadata=False)
        return {sibling.rfilename for sibling in info.siblings}
    except Exception:
        return None


def dataset_is_complete(files: set[str] | None, item: dict[str, Any]) -> bool:
    if files is None:
        return False
    expected_images = sum(counts["images"] for counts in item["counts"].values())
    expected_labels = sum(counts["labels"] for counts in item["counts"].values())
    actual_images = sum(1 for path in files if path.endswith(".jpg"))
    actual_labels = sum(1 for path in files if path.endswith(".txt"))
    forbidden = [
        path
        for path in files
        if re.search(r"(^|/)\.git(/|$)|\.cache$|\.zip$", path)
    ]
    return (
        "README.md" in files
        and "data.yaml" in files
        and actual_images == expected_images
        and actual_labels == expected_labels
        and not forbidden
    )


def model_is_complete(files: set[str] | None, item: dict[str, Any]) -> bool:
    if files is None:
        return False
    return "README.md" in files and item["filename"] in files


def upload_file_if_exists(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str | None,
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
) -> None:
    if not local_path.exists():
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def upload_dataset_batched(
    api: HfApi,
    *,
    root: Path,
    item: dict[str, Any],
) -> None:
    dataset_dir = root / item["local_dir"]
    upload_file_if_exists(
        api,
        repo_id=item["repo_id"],
        repo_type="dataset",
        local_path=dataset_dir / "data.yaml",
        path_in_repo="data.yaml",
        commit_message=f"Upload {item['title']} data config",
    )
    for extra_file in ["README.dataset.txt", "README.roboflow.txt"]:
        upload_file_if_exists(
            api,
            repo_id=item["repo_id"],
            repo_type="dataset",
            local_path=dataset_dir / extra_file,
            path_in_repo=extra_file,
            commit_message=f"Upload {item['title']} source metadata",
        )

    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            folder = dataset_dir / split / kind
            if not folder.exists():
                continue
            print(f"Uploading {item['repo_id']} {split}/{kind} ...", flush=True)
            api.upload_folder(
                repo_id=item["repo_id"],
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=f"{split}/{kind}",
                ignore_patterns=EXCLUDE_PATTERNS,
                commit_message=f"Upload {item['title']} {split}/{kind}",
            )


def upload_dataset_large_folder(
    api: HfApi,
    *,
    root: Path,
    item: dict[str, Any],
    workers: int | None,
) -> None:
    api.upload_large_folder(
        repo_id=item["repo_id"],
        repo_type="dataset",
        folder_path=str(root / item["local_dir"]),
        ignore_patterns=EXCLUDE_PATTERNS,
        num_workers=workers,
    )


def upload_all(
    root: Path,
    manifest: dict[str, Any],
    token: str | None,
    *,
    dataset_upload_mode: str,
    force: bool,
    large_folder_workers: int | None,
    cards_only: bool,
) -> None:
    api = HfApi(token=token)
    whoami = api.whoami()
    if whoami.get("name") != ACCOUNT:
        raise RuntimeError(f"Expected HF account {ACCOUNT}, got {whoami.get('name')!r}")

    for item in manifest["datasets"]:
        api.create_repo(
            repo_id=item["repo_id"],
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        existing_files = repo_file_names(api, item["repo_id"], "dataset")
        if not force and dataset_is_complete(existing_files, item):
            if cards_only:
                upload_readme(api, item["repo_id"], "dataset", dataset_card(item))
                print(f"Updated dataset card: {item['repo_id']}", flush=True)
            else:
                print(f"Skipping complete dataset repo: {item['repo_id']}", flush=True)
            continue

        upload_readme(api, item["repo_id"], "dataset", dataset_card(item))
        if cards_only:
            print(f"Updated dataset card: {item['repo_id']}", flush=True)
            continue

        if dataset_upload_mode == "large-folder":
            upload_dataset_large_folder(
                api,
                root=root,
                item=item,
                workers=large_folder_workers,
            )
        else:
            upload_dataset_batched(api, root=root, item=item)

    for item in manifest["models"]:
        api.create_repo(repo_id=item["repo_id"], private=False, exist_ok=True)
        existing_files = repo_file_names(api, item["repo_id"], None)
        if not force and model_is_complete(existing_files, item):
            if cards_only:
                upload_readme(api, item["repo_id"], None, model_card(item))
                print(f"Updated model card: {item['repo_id']}", flush=True)
            else:
                print(f"Skipping complete model repo: {item['repo_id']}", flush=True)
            continue

        upload_readme(api, item["repo_id"], None, model_card(item))
        if cards_only:
            print(f"Updated model card: {item['repo_id']}", flush=True)
            continue

        api.upload_file(
            path_or_fileobj=str(root / item["filename"]),
            path_in_repo=item["filename"],
            repo_id=item["repo_id"],
            commit_message=f"Upload {item['title']} checkpoint",
        )


def verify(api: HfApi, manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"datasets": [], "models": []}

    for item in manifest["datasets"]:
        info = api.dataset_info(item["repo_id"], files_metadata=False)
        siblings = [s.rfilename for s in info.siblings]
        forbidden = [
            path
            for path in siblings
            if re.search(r"(^|/)\.git(/|$)|\.cache$|\.zip$", path)
        ]
        results["datasets"].append(
            {
                "repo_id": item["repo_id"],
                "has_readme": "README.md" in siblings,
                "forbidden_files": forbidden,
                "file_count": len(siblings),
            }
        )

    for item in manifest["models"]:
        info = api.model_info(item["repo_id"], files_metadata=False)
        siblings = [s.rfilename for s in info.siblings]
        pt_files = [path for path in siblings if path.endswith(".pt")]
        results["models"].append(
            {
                "repo_id": item["repo_id"],
                "has_readme": "README.md" in siblings,
                "pt_files": pt_files,
                "expected_pt": item["filename"],
                "pt_ok": pt_files == [item["filename"]],
            }
        )

    return results


def validate_local_inputs(root: Path) -> None:
    missing = []
    for spec in DATASETS:
        if not (root / spec.local_dir / "data.yaml").exists():
            missing.append(str(root / spec.local_dir / "data.yaml"))
    for spec in MODELS:
        if not (root / spec.filename).exists():
            missing.append(str(root / spec.filename))
    if missing:
        raise FileNotFoundError("Missing required local inputs:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run, upload, and verify SnakeAid datasets/models on Hugging Face."
    )
    parser.add_argument("--root", default=".", help="Workspace root.")
    parser.add_argument("--upload", action="store_true", help="Create repos and upload files.")
    parser.add_argument("--verify", action="store_true", help="Verify repos on Hugging Face.")
    parser.add_argument(
        "--manifest-output",
        default="HUGGINGFACE_PUSH_MANIFEST.json",
        help="Where to write the generated manifest.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--dataset-upload-mode",
        choices=["batched", "large-folder"],
        default="batched",
        help="Dataset upload strategy. 'batched' uploads split/kind folders separately and is safer for retries.",
    )
    parser.add_argument(
        "--large-folder-workers",
        type=int,
        default=None,
        help="Optional worker count for --dataset-upload-mode large-folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even when the remote repo already appears complete.",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Only upload generated README.md cards; do not upload dataset/model payload files.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    validate_local_inputs(root)
    manifest = build_manifest(root)

    manifest_path = root / args.manifest_output
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    print(f"Datasets: {len(manifest['datasets'])}; models: {len(manifest['models'])}")
    print("Ignored model: SnakeAI_5k_120e.pt")

    if args.upload:
        if not args.token:
            raise RuntimeError("HF token is required for --upload. Set HF_TOKEN or pass --token.")
        upload_all(
            root,
            manifest,
            args.token,
            dataset_upload_mode=args.dataset_upload_mode,
            force=args.force,
            large_folder_workers=args.large_folder_workers,
            cards_only=args.cards_only,
        )
        print("Upload complete.")

    if args.verify:
        if not args.token:
            raise RuntimeError("HF token is required for --verify. Set HF_TOKEN or pass --token.")
        api = HfApi(token=args.token)
        results = verify(api, manifest)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
