# SOTA Engineering Paradigms (2026 Edition)

**Purpose:** To ensure generated code adheres to modern industry standards, ensuring efficiency, readability, and reproducibility.

## 1. General PyTorch Standards (Mandatory)

* **Precision:** Always use **Automatic Mixed Precision (AMP)** (`torch.amp.autocast`). Do not use full FP32 unless necessary for numerical stability in specific loss functions.
* **Optimization:**
  * Use `torch.optim.AdamW` instead of `Adam`.
  * Always use a Learning Rate Scheduler (e.g., `CosineAnnealingLR` with Warmup) rather than constant LR.
* **Data Loading:**
  * `num_workers` must be configurable (default to `os.cpu_count()`).
  * Use `pin_memory=True` for GPU training.
* **Banned Patterns (Anti-Patterns):**
  * ❌ `Variable(tensor)` (Deprecated since 0.4.0)
  * ❌ `.data` (Use `.detach()`)
  * ❌ Hardcoded `cuda:0` (Use `device` object passed via config)
  * ❌ Manual loop for gradient accumulation (Use logic within the Trainer loop properly)

## 2. Domain-Specific Guidelines

### Computer Vision (CV)

* **Backbones:** Do not write ResNet/ViT from scratch unless the paper introduces a *new* architecture modification. Use **`timm` (PyTorch Image Models)** to instantiate backbones.
  * *Code:* `timm.create_model('resnet50', pretrained=False, num_classes=K)`
* **Augmentation:** Use **`torchvision.transforms.v2`** or **`albumentations`**. Do not implement raw numpy flips manually.

### NLP / LLM

* **Framework:** Prefer **HuggingFace `Trainer`** abstraction over raw loops for standard fine-tuning.
* **Tokenization:** Always save the `tokenizer` alongside the model checkpoint.
* **Acceleration:** Use `accelerate` library for device placement if writing raw loops.

### Reinforcement Learning (RL)

* **Vectorization:** Use `Gymnasium`'s `VectorEnv` for parallel environment stepping.
* **Buffers:** Use efficient Replay Buffer implementations (e.g., from `stable-baselines3` or `CleanRL`), do not use a Python list of tuples `[(s,a,r,s')]`.

## 3. Logging & Artifacts

* **Logging:** Use `wandb` (Weights & Biases) or `tensorboard`. Never rely solely on console `print`.
* **Seeds:** Must implement a global `seed_everything(seed)` function covering `torch`, `numpy`, and `random`.
