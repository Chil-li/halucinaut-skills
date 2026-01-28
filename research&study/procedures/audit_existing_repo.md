# Procedure: Existing Repository Audit (Branch A)

此流程仅在检测到有效的 GitHub 仓库或完整代码包时触发。
**目标：** 解决 "Works on my machine" 问题，检测代码腐烂，并验证代码是否忠实于论文。

---

## Step 1: 时效性与依赖分析 

**1.1 确立时间基准**

* 获取当前系统日期 (Current Date)。
* 检查代码库的 `Last Commit Date` 或文件修改时间。
* **计算时间差 **：如果代码库超过 18 个月未更新，自动标记为 **[HIGH RISK: Code Rot]**。

**1.2 依赖文件解析**
扫描 `requirements.txt`, `environment.yml`, `setup.py` 或 `Dockerfile`。执行以下检查：

* **版本锁定检查**：

  * 如果版本未锁定 (e.g., `numpy`) -> 警告：可能因 API 变动导致崩溃（如 `np.float` 被移除）。
  * 如果版本过旧 (e.g., `torch==1.7.0`) -> 警告：可能与当前主流 GPU (e.g., RTX 4090/5090) 的 CUDA 架构不兼容。
* **API 废弃检测 **：
  搜索代码中是否包含已废弃的写法。如果发现，必须在报告中指出并提供迁移建议：

  * `torch.autograd.Variable` (已废弃) -> 应直接使用 `torch.Tensor`
  * `tensorflow.contrib` (TF 1.x) -> 必须标记为需要重构
  * `np.bool`, `np.int` (NumPy 1.20+ 已移除) -> 应建议替换为 Python 原生类型

**1.3 输出安装策略**
基于上述分析，生成一个**当前可用**的 `conda` 安装脚本。

* *如果依赖过旧*：尝试推荐兼容当前 CUDA 版本的最低 PyTorch 版本，或者建议使用 Docker 容器。

---

## Step 2: 论文-代码一致性审计

**严禁**默认假设代码就是论文的完美实现。必须进行双盲对比：

**2.1 超参数对齐**
提取 `config.py`, `args.py` 或 `main.py` 中的默认参数，与论文正文进行对比。重点关注：

| 检查项                  | 常见差异风险                                      |
| :---------------------- | :------------------------------------------------ |
| **Batch Size**    | 代码中常为了适配旧显卡而调小，导致收敛性差异      |
| **Learning Rate** | 代码中可能包含了论文未提及的 Warmup 或 Decay 策略 |
| **Epochs**        | 代码可能为了快速 Demo 而设置极小的 Epoch 数       |
| **Random Seed**   | 检查是否有固定的 Seed。若无，标记为不可复现风险   |

**2.2 隐性 Trick 挖掘 **
扫描 `forward()` 函数和 `loss` 计算部分，查找论文中未提及的操作：

* 是否有额外的 Dropout？
* 是否使用了特殊的 Initialization 方法？
* Data Loader 中是否有论文未提及的 Augmentation（如 Mixup, CutMix）？
  * *Action*: 如果发现，必须在审计报告中高亮：“代码包含论文未提及的 [X] 技术，这可能是复现关键。”

---

## Step 3: 执行环境沙箱模拟

在生成运行指令前，进行静态路径分析：

**3.1 硬编码路径检查 **
扫描代码中是否存在绝对路径 (e.g., `/home/user/data/imagenet`)。

* *Action*: 如果存在，警告用户必须修改，并给出修改位置（行号）。

**3.2 硬件资源匹配**
检查 `config` 中的 `num_workers` 和 `batch_size`。

* 如果检测到用户是单卡环境但代码默认多卡 (`DataParallel` / `DDP`)，自动生成修改建议（如设置 `CUDA_VISIBLE_DEVICES` 或修改 `device_ids`）。

---

## Step 4: 最终输出生成

生成一份 **"复现就绪包 "**：

1. **修正后的环境安装脚本** (修复了过时库)。
2. **差异警告表** (Paper vs Code)。
3. **One-Liner 启动命令** (e.g., `python train.py --data_dir ./data --batch_size 16`)，确保参数已适配用户当前环境。
