# Procedure: Missing Flow Reconstruction (Branch B)

此流程在无代码或仅有部分代码（如仅 Inference 脚本）时触发。
**核心目标：** 基于论文逻辑，利用当前（{{CURRENT_DATE}}）的行业主流范式（SOTA Paradigms）构建可维护的训练/评估框架。

---

## Step 1: 行业范式对齐 (Paradigm Alignment)

在编写任何具体逻辑前，首先根据论文领域确定代码的“骨架”。**严禁**使用低效的、非标准化的原生 PyTorch 循环，除非论文算法极其特殊。

| 领域                            | 推荐范式                                          | 理由                                                                                        |
| :------------------------------ | :------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **LLM / NLP**             | **HuggingFace `Trainer` / `Accelerate`** | 自动处理 FP16/BF16, Gradient Accumulation, DDP。                                            |
| **Computer Vision**       | **`timm` style / PyTorch Lightning**       | 利用 `timm` 的预训练骨干和数据增强 Pipeline。                                             |
| **RL**                    | **CleanRL / Ray Rllib**                      | RL 极其敏感，必须使用经过验证的 Env 交互骨架。                                              |
| **General Deep Learning** | **PyTorch (Standard Boilerplate)**           | 必须包含 `Dataset`, `DataLoader`, `Model`, `Optimizer`, `Loop` 的清晰类结构分离。 |

**执行动作：**
向用户声明将采用的框架（例如：“检测到这是 Vision Transformer 相关论文，将基于 `timm` 风格构建代码以保证高性能”）。

---

## Step 2: 逻辑映射与填补 

**2.1 数学公式 -> 代码函数**

* 提取论文中的核心 Equation（如自定义 Loss 或 Attention Mask）。
* 编写代码时，Docstring 必须引用公式。
  * *Example:* `def computed_loss(x): # Implements Eq. 4 in Section 3.2`

**2.2 缺失信息的“智性诚实”处理**
当遇到论文未提及的超参数（Gap）时，按以下优先级处理：

1. **Priority 1 (Inferred):** 从上下文或图表中推断（如根据架构图推断 Stride）。
2. **Priority 2 (Standard):** 使用该领域 {{CURRENT_DATE}} 的默认标准（如 AdamW default betas）。
3. **Priority 3 (Alert):** 如果是关键参数（如 Loss 的权重 $\lambda$），必须在代码中设置为 `None` 或 `TODO`，并抛出 `NotImplementedError`，强制用户确认。

---

## Step 3: 强制性 Mock 测试 

在生成完整的训练脚本前，**必须**先生成并验证一个微型的 `mock_test.py`。这是 Branch B 的质量阀门。

**Mock 测试清单：**

1. **Shape Consistency:**
   * 生成随机 Input Tensor (e.g., `torch.randn(2, 3, 224, 224)`).
   * 运行 Model `forward()`.
   * 断言 Output Shape 符合预期 (e.g., `assert out.shape == (2, num_classes)`).
2. **Loss Reduction:**
   * 运行 Loss Function。
   * 验证 Loss 是否输出了一个 scalar，且没有 `NaN`。
3. **Gradient Check:**
   * 执行 `loss.backward()`.
   * 检查主要层的 `param.grad` 是否不为 None。

*如果 Mock 测试不通过，严禁生成后续代码。*

---

## Step 4: 完整工程构建 

通过 Mock 测试后，生成最终工程文件结构：

1. **`config.py`**:
   * 所有的 Magic Numbers (Batch Size, LR, Channels) 全部移入此文件。
   * 包含 `get_device()` 逻辑（自动适配 MPS/CUDA）。
2. **`model.py`**:
   * 定义模型架构。如果引用了预训练模型（如 ResNet），必须指定 `weights=` 参数，避免使用过时的 `pretrained=True`。
3. **`train.py`**:
   * 实现 Training Loop。
   * 必须包含 `tqdm` 进度条。
   * 必须包含 `logging`（不再单纯使用 `print`）。
   * 必须包含 Checkpoint 保存逻辑 (`torch.save`)。

---

## 5. 交付物检查 

在输出代码块之前，自检以下“毒点”：

* [ ] 是否使用了硬编码的路径？ (应使用 `argparse`)
* [ ] 是否使用了过时的 `autograd` 语法？
