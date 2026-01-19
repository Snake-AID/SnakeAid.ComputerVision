# Project overview

![SnakeAid banner](img/SnakeAI.png)

SnakeAid Computer Vision focuses on training and packaging YOLO models for snake detection and related computer vision tasks. The repo covers data labeling, training workflows, experiment tracking, and packaging models for deployment.

## Tech stack snapshots

![Trainer stack](img/TechStackTrainer.png)
![Deployment stack](img/TechStackDeployment.png)

## Pipeline at a glance

```mermaid
flowchart TD
    A[Roboflow] --> B[Ultralytics YOLO]
    B --> C[Colab / SageMaker]
    C --> D[ClearML (Tracking and Registry)]
    D --> E[ONNX / PyTorch Models]
    E --> F[Neural Magic (CPU Inference)]
```

## Focus areas

- Keep dataset labeling and versioning repeatable with Roboflow exports in YOLO format.
- Train YOLO models in notebook-friendly environments (Colab, SageMaker) with reproducible configs.
- Track experiments and artifacts centrally in ClearML.
- Package models for downstream inference (GPU or CPU via ONNX and Neural Magic).
