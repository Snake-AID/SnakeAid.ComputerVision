# Training frameworks and environments

## Ultralytics YOLO

YOLO provides the core detection and segmentation architecture along with training, validation, and export workflows. It balances real-time performance, accuracy, and flexible deployment targets (GPU, CPU, ONNX).

## Google Colab

Primary training environment for fast iteration with GPUs, notebooks, and quick dataset validation.

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <td style="border: none; text-align: center;"><img src="img/Techstack/Colab/Training.png" alt="Training" style="max-width: 100%; height: auto;"><br><small>Training</small></td>
    <td style="border: none; text-align: center;"><img src="img/Techstack/Colab/Evaluate.png" alt="Evaluate" style="max-width: 100%; height: auto;"><br><small>Evaluate</small></td>
  </tr>
  <tr>
    <td style="border: none; text-align: center;"><img src="img/Techstack/Colab/ColabPro.png" alt="Colab Pro" style="max-width: 100%; height: auto;"><br><small>Colab Pro</small></td>
    <td style="border: none; text-align: center;"><img src="img/Techstack/Colab/Runtimes.png" alt="Runtimes" style="max-width: 100%; height: auto;"><br><small>Runtime options</small></td>
  </tr>
</table>

Use cases:

- Initial experiments and hyperparameter tuning.
- Validating dataset quality quickly.
- Generating artifacts for analysis.

## Amazon SageMaker (experimental)

Secondary environment to compare notebook-driven training with managed ML workflows and to study production-grade infrastructure. Not the primary training path, but useful for understanding scaling and orchestration options.
