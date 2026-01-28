# 论文复现完备性审计报告 

**审计对象 :** {{PAPER_TITLE}}
**当前日期:** {{CURRENT_DATE}} (用于评估依赖库时效性)
**审计分支:** {{BRANCH_NAME}} (Branch A: Code Audit / Branch B: Reconstruction)

---

## 1. 核心复现要素检查

*请基于文中明确陈述的事实填写，若需推断请标记 [Inferred]*

| 要素                         | 状态                           | 详细说明/数值                        | 来源位置 (Section/Eq) |
| :--------------------------- | :----------------------------- | :----------------------------------- | :-------------------- |
| **Model Architecture** | [ ] 明确 / [ ] 模糊 / [ ] 缺失 | (e.g., ResNet-50 w/ modified stride) |                       |
| **Objective / Loss**   | [ ] 明确 / [ ] 模糊 / [ ] 缺失 | (e.g., CrossEntropy + 0.1 * L1)      |                       |
| **Optimization**       | [ ] 明确 / [ ] 模糊 / [ ] 缺失 | (Optimizer, Base LR, Schedule)       |                       |
| **Batch Size**         | [ ] 明确 / [ ] 模糊 / [ ] 缺失 |                                      |                       |
| **Weight Init**        | [ ] 明确 / [ ] 模糊 / [ ] 缺失 | (关键！若缺失极难复现)               |                       |
| **Data Preprocessing** | [ ] 明确 / [ ] 模糊 / [ ] 缺失 | (Normalization stats, Augmentation)  |                       |

---

## 2. 差异与风险分析 

### 2.1 依赖与时效性风险

*基于当前日期 {{CURRENT_DATE}} 评估*

* **主流框架兼容性**: [ ] High (PyTorch 2.x+) / [ ] Medium / [ ] Low (Legacy Code)
* **过时 API 警告**:
  * (列出代码或论文中提到的已废弃 API，例如 `torch.autograd.Variable`)
  * (列出可能存在的 CUDA 版本冲突)

### 2.2 逻辑一致性风险

*如果是 Branch A，对比 Paper 与 Code；如果是 Branch B，检查公式推导闭环*

* **[CRITICAL]**: (在此列出导致无法复现的致命矛盾，如 Code 中使用了 Paper 未提及的 Trick)
* **[WARNING]**: (在此列出参数不一致，如 LR Paper=1e-3, Code=1e-4)

---

## 3. 资源预估 

* **Params**: ~{{PARAM_COUNT}}
* **Estimated VRAM (Training)**: {{VRAM_GB}} GB (Based on Batch Size {{BATCH_SIZE}} & FP32/FP16)
* **Estimated Time**: {{TIME_ESTIMATE}} (on standard GPU)

---

## 4. 审计结论

*请选择一项并简述理由*

- [ ] **GREEN (Ready to Implement)**: 信息完备，风险可控。
- [ ] **YELLOW (Proceed with Caution)**: 缺少非关键信息（如具体 Init），将使用行业标准 {{STANDARD_USED}} 代替。
- [ ] **RED (Blocker Detected)**: 缺少关键信息/存在逻辑硬伤，无法进行科学复现。请用户补充资料。

**下一步建议 :**
{{ACTION_ITEM}}
