# Mission: CSI-Bench-SSL

## Purpose
The goal of this repository is to investigate **Transfer Learning** on **Channel State Information (CSI)** data. 
Specifically, we focus on **Self-Supervised Learning (SSL)** pretext tasks (like VQ-CPC) to learn robust representations that generalize across:
- **Environments** (e.g., Room A -> Room B)
- **Devices** (e.g., Rx1 -> Rx2)
- **Users** (e.g., Person 1 -> Person 2)

Key applications include **Breathing Detection** and **Fall Detection**.

## Core Components
1.  **Pretext Tasks (SSL):** Training models on unlabeled data to learn features (Frame-wise or Window-wise).
    -   Script: `scripts/train_ssl.py`
    -   Goal: Learn a robust `encoder`.
2.  **Baselines (Supervised):** Training standard classifiers (MLP, ResNet, ViT) from scratch on labeled data.
    -   Script: `scripts/train_supervised.py`
    -   Goal: Establish performance benchmarks without SSL.
3.  **Transfer Learning (Fine-tuning):** Leveraging the pre-trained `encoder` for downstream tasks.
    -   **Linear Probing:** Freeze encoder, train a linear classifier on top. (Evaluates representation quality).
    -   **Fine-tuning:** Train both encoder (low LR) and classifier. (Maximizes performance).
    -   *Note:* Currently requires adapting `train_supervised.py` to load pre-trained encoders.

## Transfer Protocols
- **Cross-Environment:** Train on Room A, Test on Room B.
- **Cross-User:** Train on Users 1-N, Test on User N+1.
- **Cross-Device:** Train on Rx1, Test on Rx2.


## Agent Guidelines
- **Consistency:** Maintain a clear separation between Pretext and Downstream tasks.
- **Reproducibility:** Ensure experiments are easily reproducible with config files or clear arguments.
- **Modularity:** Avoid duplicating logic between supervised and self-supervised training loops. Use shared engines where possible.

## How to Run
Use `pixi` to manage environments and dependencies.

### macOS (Apple Silicon / MPS)
```bash
pixi run -e mps python scripts/train_ssl.py --data_dir data/csi-bench-ssl/csi-bench-ssl --task FallDetection ...
```

### Linux (CUDA)
```bash
pixi run python scripts/train_ssl.py --data_dir data/csi-bench-ssl/csi-bench-ssl --task FallDetection ...
```
