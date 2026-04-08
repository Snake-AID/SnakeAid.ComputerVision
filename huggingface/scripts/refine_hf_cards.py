from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from push_snakeaid_to_hf import (
    ACCOUNT,
    build_manifest,
    dataset_card,
    model_card,
    upload_readme,
    validate_local_inputs,
    verify,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate and upload only SnakeAid Hugging Face README cards."
    )
    parser.add_argument("--root", default=".", help="Workspace root.")
    parser.add_argument("--verify", action="store_true", help="Verify repos after updating cards.")
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
        "--force",
        action="store_true",
        help="Kept for CLI compatibility; card refinement always updates README.md only.",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        default=[],
        help="Only update the given repo id. Can be passed more than once.",
    )
    args = parser.parse_args()

    if not args.token:
        raise RuntimeError("HF token is required. Set HF_TOKEN or pass --token.")

    root = Path(args.root).resolve()
    validate_local_inputs(root)
    manifest = build_manifest(root)

    manifest_path = root / args.manifest_output
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    print("Updating README cards only; dataset/model payload files will not be uploaded.")

    api = HfApi(token=args.token)
    whoami = api.whoami()
    if whoami.get("name") != ACCOUNT:
        raise RuntimeError(f"Expected HF account {ACCOUNT}, got {whoami.get('name')!r}")

    selected = set(args.repo_id)

    for item in manifest["datasets"]:
        if selected and item["repo_id"] not in selected:
            continue
        print(f"Updating dataset card: {item['repo_id']} ...", flush=True)
        api.create_repo(
            repo_id=item["repo_id"],
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        upload_readme(api, item["repo_id"], "dataset", dataset_card(item))
        print(f"Updated dataset card: {item['repo_id']}", flush=True)

    for item in manifest["models"]:
        if selected and item["repo_id"] not in selected:
            continue
        print(f"Updating model card: {item['repo_id']} ...", flush=True)
        api.create_repo(repo_id=item["repo_id"], private=False, exist_ok=True)
        upload_readme(api, item["repo_id"], None, model_card(item))
        print(f"Updated model card: {item['repo_id']}", flush=True)

    print("Card update complete.")

    if args.verify:
        api = HfApi(token=args.token)
        results = verify(api, manifest)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
