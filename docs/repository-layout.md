# Repository layout

This repository is organized by workflow stage so large notebooks, documentation,
and local helper tools do not compete for space at the root.

```text
.
|-- docs/                         # Docsify documentation site
|-- notebooks/
|   |-- training/                 # YOLO training notebook history
|   `-- validation/               # Dataset and label verification notebooks
|-- tools/
|   `-- colab-shutdown-webhook/   # Local webhook helper for Colab completion
`-- README.md                     # Repo entrypoint
```

## docs/

Docsify site for the project. `docs/README.md` is the documentation landing
page, and `docs/_sidebar.md` controls navigation.

Serve locally with:

```bash
npx docsify-cli@latest serve docs
```

## notebooks/training/

Historical YOLO training notebooks. These are kept as notebook artifacts rather
than being split into Python modules because the training workflow is Colab-first
and experiment-oriented.

## notebooks/validation/

Dataset and label verification notebooks. These are separated from training runs
so quality checks are easier to find.

## tools/colab-shutdown-webhook/

Windows helper for exposing a local webhook through ngrok and shutting down the
machine after a Colab training run calls back.
