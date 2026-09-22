# 从克隆仓库到训练与测试：SideNet 操作指南

本文面向首次使用项目的用户，按顺序完成：安装环境 → 生成数据 → 划分训练/验证/测试集 → 试跑 → 正式训练 → 最终测试。
命令使用 Linux / WSL 的 Bash，推荐 Python 3.12。CPU 可以跑通全部流程，正式训练可使用 NVIDIA GPU。

本教程采用 **DGCNN + 真实赛道地图与虚拟赛道混合数据 + 赛道家族隔离**。
这里的“真实赛道数据”也是基于真实地图的感知仿真，不是实车 LiDAR 采集数据。
模型输入为一帧锥桶位置，输出每个锥桶的左右侧别；本流程不训练点云目标检测器。

## 1. 克隆两个仓库

在你希望保存项目的位置执行：

```bash
mkdir -p lidar-cone-perception
cd lidar-cone-perception
git clone https://github.com/orangeorangehc/sidenet.git SideNet
git clone https://github.com/orangeorangehc/bitfsd-generator.git bitfsd-generator
cd SideNet
```

请保留下面的相邻目录结构，尤其是 `SideNet` 的大小写：

```text
lidar-cone-perception/
├── SideNet/
│   ├── prepare_mixed_split.py
│   ├── train_sidenet.py
│   ├── infer.py
│   └── configs/
└── bitfsd-generator/
    ├── collect_multiseed.py
    ├── config/
    └── data/                    # 已纳入 Git 的赛道地图 YAML
```

使用 fork 时替换仓库 URL；复现某次实验时，两边都切换到该实验记录的 commit。
如果缺少上述脚本，请先核对分支和版本。
`.venv/`、生成的 `output/`、训练的 `runs/` 和权重文件均被 Git 忽略，**克隆后需要自行安装和生成**。
已有文档中“数据已生成”“环境已验证”等内容是作者机器的历史记录。

## 2. 安装一个共享 Python 环境

以下命令在 `SideNet/` 中执行。生成器的批量采集流程与训练器都使用这个环境，避免混用系统 Python：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.4.4 PyYAML==6.0.3
```

然后按硬件选择**一条** PyTorch 安装命令：

```bash
# CPU：适合先跑通流程
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU：需要支持 CUDA 12.8 runtime 的驱动
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

如果 GPU/驱动不适配上述 wheel，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
选择兼容版本，并记录实际版本。预编译 wheel 通常不要求另装 CUDA Toolkit，但需要兼容的 NVIDIA 驱动。
当前数据生成、划分、训练和本文评估流程只需要 NumPy、PyYAML、PyTorch；
若还要运行生成器 Web 页面或可视化，再按生成器 README 安装 `pyproject.toml` 中的完整依赖。

检查正在使用的解释器和 GPU：

```bash
python -c "import sys, torch, numpy, yaml; print(sys.executable); print('torch:', torch.__version__, 'numpy:', numpy.__version__, 'yaml:', yaml.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

`CUDA: False` 时先用 `--device cpu`。需要 GPU 时，检查 `nvidia-smi`、驱动、wheel 和 Python 环境。
后续命令显式指定 `.venv/bin/python`，新开终端也不依赖环境是否激活。

## 3. 生成训练用数据

从 `SideNet/` 切换到生成器：

```bash
cd ../bitfsd-generator
../SideNet/.venv/bin/python collect_multiseed.py \
  --config config/perceive_mixed.yaml
```

配置 `bitfsd-generator/config/perceive_mixed.yaml` 的主要内容：

| 项目 | 默认设置 | 含义 |
|---|---|---|
| 真实地图 | `real_tracks` 中 14 个文件 | 包含同一家族的不同版本 |
| 虚拟预设 | simple_oval、tight_oval、hairpin、triangle | 4 种固定几何 |
| 观测 seed | 42、43、44 | 改变噪声和漏检等观测，不增加独立赛道几何 |
| LiDAR 范围/视场 | 0.5–30 m / 180° | 根据实际传感器调整 |
| ego 采样间距 | 5 m | 间距越小，相邻帧通常越相似 |
| 位置噪声 / 漏检率 | XY 0.10 m、Z 0.03 m / 0.10 | 感知仿真参数 |
| 坐标 / 反向采集 | `ego` / `flip: false` | 本多 seed 配方要求此设置 |

成功后输出：

```text
bitfsd-generator/output/sidenet_data_mixed_ego/
├── dataset_manifest.yaml
├── FSE22_s42/
│   ├── metadata.yaml
│   ├── cloud_0.txt
│   └── ...
├── FSE22_s43/
├── simple_oval_s42/
└── ...
```

每个非空 `cloud_N.txt` 表示一帧，每行一个锥桶，共 9 列：

```text
x y z dx dy dz heading score class_name
```

坐标以米为单位，`+x` 向前、`+y` 向左；标签为 `Cone_Left` / `Cone_Right`。
`side_semantics: track_global` 表示相对于赛道规定行驶方向的左右边界身份。
二分类训练导出不包含虚警；即使配置 `fp_rate`，也不会把虚警作为带真实左右标签的样本写入。
合成 score 为常数，推荐保留 `input_mode: xyz`。

默认配置的历史参考规模是 **54 个目录、2,535 帧、73,394 个锥桶观测**。
数量用于排查配置差异，实际以生成结果为准；相同 seed 还需要相同代码、地图、预设、依赖与处理顺序才能复现。
根 manifest 和每条赛道的 metadata 必须与帧文件一起保存，不能只复制 txt。

**重复执行会拒绝覆盖已有输出目录。** 新实验用一个新目录：

```bash
# 仍在 bitfsd-generator/；这是可选的新版本示例
../SideNet/.venv/bin/python collect_multiseed.py \
  --config config/perceive_mixed.yaml \
  --output output/sidenet_data_mixed_ego_v2
```

主流程后续使用默认目录；如果选择了 `_v2`，划分时也要指定它。

## 4. 划分训练集、验证集和测试集

```bash
cd ../SideNet
.venv/bin/python prepare_mixed_split.py
```

脚本读取 `configs/mixed_split.yaml`，检查数据契约、标签、seed 完整性和家族隔离，生成或覆盖：

| 文件 | 用途 |
|---|---|
| `configs/dgcnn_mixed.yaml` | 训练配置；只允许加载训练和验证目录 |
| `splits/mixed_ego.json` | 三个分区的帧 ID、统计、家族与数据 SHA-256 |

划分不移动或复制原始数据。训练程序根据生成的 `include_dirs` 和 `val_groups` 复现训练/验证划分；
它不会直接读取三分区 JSON，因此更换数据或划分策略后必须重新运行准备脚本。
该脚本从 `configs/dgcnn.yaml` 生成配置，会覆盖你对旧 `dgcnn_mixed.yaml` 的手动调参；请先划分，再复制配置调参。

默认家族分配如下：

| 来源 | 训练 | 验证 | 测试 |
|---|---|---|---|
| 真实地图 | FSE22、FSE23、FSE24、FSG19、FSG23、FSI24、FSO20、FSS19 | FSCZ24、FSG21 | FSG24、FSS22 |
| 虚拟预设 | simple_oval、tight_oval、hairpin | triangle | 无 |

默认配置的历史统计：

| 分区 | 帧数 | 锥桶观测数 | 用途 |
|---|---:|---:|---|
| 训练 | 1,686 | 51,075 | 更新参数 |
| 验证 | 471 | 11,087 | 选择 epoch、调参和模型方案 |
| 测试 | 378 | 11,232 | 方案冻结后最终评估 |

同一赛道的相邻帧、全部 seed 以及派生版本必须留在同一分区。
`FSE22_test` 文件名中的 `_test` 不决定用途，它属于 FSE22 训练家族；`FSS22_V1/V2` 都属于测试家族。
不要随机按 txt 文件做 8:1:1 切分，否则同源观测可能跨集，导致评估偏高。

自定义数据时先修改采集配置，再修改划分策略中的 `expected_seeds`、`real`、`synthetic`。
每个家族必须且只能出现在一个分区；策略列出的家族必须与数据实际家族完全一致。
当前准备脚本要求三个分区均非空、均有左右两类，且最终测试只包含真实地图。
若新增同源地图却使用不同名称，自动去除 `_s数字`、`_V数字`、`_test`、`_flip` 后缀无法识别其关系，
需要统一命名或扩展 `canonical_track_group()`，确保它们不会跨集。

例如，为 `_v2` 数据生成独立配置与清单：

```bash
.venv/bin/python prepare_mixed_split.py \
  --data-dir ../bitfsd-generator/output/sidenet_data_mixed_ego_v2 \
  --policy configs/mixed_split.yaml \
  --config-output configs/dgcnn_mixed_v2.yaml \
  --manifest-output splits/mixed_ego_v2.json
```

之后训练要使用 `_v2.yaml` 和新的 `--save-dir`，最终测试也要使用 `_v2.json`。
指定了 `val_groups` 时，改变 `val_ratio` 或 `--split-seed` 不会改变家族归属。

## 5. 先完成一个 epoch 的试跑

以下命令均在 `SideNet/` 执行。用独立目录保存试跑结果：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u train_sidenet.py configs/dgcnn_mixed.yaml \
  --epochs 1 --device cpu --save-dir runs/dgcnn_mixed_smoke
```

确认启动日志中的数据路径、`frame=ego`、`side=track_global`、训练/验证数量与第 4 步一致，
末尾出现 best/last checkpoint 的保存路径。
试跑只证明加载、训练、验证和保存可执行，不代表模型已训练充分。
若要预检 GPU，把 `--device cpu` 改成 `--device cuda`，并采用下一节的 CUDA 环境变量。

## 6. 开始正式训练

默认配置为 DGCNN、xyz 输入、300 epochs、batch size 16、学习率 0.001、weight decay 0.01。
按验证集 `macro_track_accuracy` 选择最佳 epoch。

GPU 命令：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u train_sidenet.py configs/dgcnn_mixed.yaml \
  --device cuda --save-dir runs/dgcnn_mixed_run01
```

CPU 命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u train_sidenet.py configs/dgcnn_mixed.yaml \
  --device cpu --save-dir runs/dgcnn_mixed_run01
```

选择其中一条。`CUBLAS_WORKSPACE_CONFIG` 用于满足已启用的确定性 CUDA 运算要求。
远程长训练可在 `tmux` 会话中启动，避免 SSH 断开中止进程。

常用覆盖参数为 `--epochs`、`--lr`、`--batch-size`、`--seed`、`--device`、`--save-dir`。
例如另一次实验可加 `--epochs 100 --lr 0.0005 --batch-size 8 --seed 7`，同时换一个保存目录。
`--seed` 改变模型初始化与训练增强，不改变这里固定的家族划分。
配置中的数据路径与保存路径相对于 **SideNet 项目根目录**，不是 `configs/`。

训练产物：

```text
runs/dgcnn_mixed_run01/
├── dgcnn.pth             # 验证指标最优 epoch 的权重
├── dgcnn.last.pth        # 最后一 epoch 的权重
├── split_manifest.json  # 本次实际训练/验证帧与 SHA-256
└── metrics.jsonl        # 每轮训练及验证指标
```

**当前训练器在所有 epochs 正常结束后才写入 best/last 权重。** 每轮会写 metrics，
但中途退出可能没有本次权重；目前没有 `--resume`，last checkpoint 也不是含 optimizer/scheduler 状态的完整续训存档。
不具备自动 early stopping。请合理设置轮数，并保持进程运行。
复用保存目录会清空旧 metrics、重写 split，并在结束时覆盖权重，因此每次实验使用新目录。

观察验证集整体 accuracy、balanced accuracy、macro F1、`macro_track_accuracy`、`worst_track_accuracy`
以及 FSCZ24/FSG21 的逐赛道表现。当前 per-track 指标按目录统计，包含 `_s42` 等 seed 后缀，
并不是先合并赛道家族再平均。混合总体得分可能受较容易的虚拟赛道影响。
训练准确率持续上升但真实验证表现下降时，应在验证集上调整轮数、正则化或数据多样性。
训练指标包含增强和训练模式，与验证指标的差距也不能完全归因于过拟合。

## 7. 复核验证结果

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python infer.py \
  --ckpt runs/dgcnn_mixed_run01/dgcnn.pth \
  --data-dir ../bitfsd-generator/output/sidenet_data_mixed_ego \
  --device cpu --json
```

有 GPU 时可以改为 `--device cuda`。默认校验数据 SHA-256，并只评估 checkpoint 记录的验证帧。
`--json` 前仍可能输出加载提示，因此重定向后的整份 stdout 不是纯 JSON 文件。
`--all-frames` 也只会评估 checkpoint 允许加载的目录，不会纳入被排除的测试目录。
不要用 `--allow-dataset-mismatch` 绕过正式评估的数据一致性检查。

## 8. 方案冻结后评估独立测试集

现有 `infer.py` 没有 `--split test` 接口。下面调用已有 Python API，先检查完整数据、三个分区、
训练时的实际划分与 SHA-256，再只评估 test 帧。**在模型、超参数与阈值确定后执行**，不要根据测试结果继续挑选模型。

在 `SideNet/` 执行；自定义实验只需对应修改开头的 `split_path`、`ckpt_path` 和 `device`。
数据路径取自划分清单，仍相对于当前 `SideNet/` 目录。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python - <<'PY'
import json
from pathlib import Path

from infer import SideNetPredictor, evaluate_labeled
from sidenet_data import canonical_track_group, dataset_fingerprint, load_frames

split_path = Path("splits/mixed_ego.json")
ckpt_path = Path("runs/dgcnn_mixed_run01/dgcnn.pth")
device = "cpu"  # 有 GPU 时可改为 "cuda"

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

manifest = json.loads(split_path.read_text())
samples, _ = load_frames(
    manifest["data_dir"],
    expected_coordinate_frame=manifest["coordinate_frame"],
    expected_side_semantics=manifest["side_semantics"],
)
check(dataset_fingerprint(samples) == manifest["dataset_sha256"], "完整数据 SHA-256 不匹配")
parts = {}
seen_ids, seen_groups = set(), set()
for name in ("train", "validation", "test"):
    info = manifest["partitions"][name]
    wanted = set(info["frame_ids"])
    part = [s for s in samples if s.sample_id in wanted]
    found = {s.sample_id for s in part}
    groups = {canonical_track_group(s.track) for s in part}
    check(bool(part) and found == wanted and len(part) == len(info["frame_ids"]), f"{name} 帧缺失或重复")
    check(not (found & seen_ids or groups & seen_groups), f"{name} 与其他分区泄漏")
    check(dataset_fingerprint(part) == info["dataset_sha256"], f"{name} SHA-256 不匹配")
    seen_ids.update(found)
    seen_groups.update(groups)
    parts[name] = part
check(seen_ids == {s.sample_id for s in samples}, "存在未划分的帧")

predictor = SideNetPredictor(ckpt_path, device)
for key in ("coordinate_frame", "side_semantics"):
    check(predictor.data_config[key] == manifest[key], f"checkpoint {key} 不匹配")
actual = predictor.checkpoint["split_manifest"]
for partition, key in (("train", "train_frames"), ("validation", "val_frames")):
    check(set(actual[key]) == {s.sample_id for s in parts[partition]}, f"checkpoint {partition} 划分不匹配")
trainval_hash = dataset_fingerprint(parts["train"] + parts["validation"])
check(trainval_hash == actual["dataset_sha256"] == manifest["train_validation_sha256"], "训练数据 SHA-256 不匹配")

metrics = evaluate_labeled(predictor, parts["test"], threshold=0.0)
report = {
    "evaluation_scope": "independent_test",
    "checkpoint": str(ckpt_path),
    "split_manifest": str(split_path),
    "test_dataset_sha256": manifest["partitions"]["test"]["dataset_sha256"],
    "threshold": 0.0,
    "metrics": metrics,
}
output = ckpt_path.parent / "test_metrics.json"
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"Saved {output}")
PY
```

默认数据应只评估 378 帧。这里阈值为 0，没有拒识，
`accuracy_with_abstentions_as_errors` 就是逐锥桶 accuracy。左右类别 ID 为 0/1，混淆矩阵行为真实类别、列为预测类别。
若业务需要置信度拒识，先在验证集确定阈值，再固定阈值进行测试并报告 coverage。
测试集只含真实地图，最终结果说明的是这些保留地图上的仿真泛化，不能直接视为实车或闭环性能。

## 9. 常见问题

| 现象 | 处理方法 |
|---|---|
| `No module named torch/yaml/numpy` | 核对是否使用 `SideNet/.venv/bin/python`，用同一解释器的 `-m pip` 安装依赖 |
| 生成器提示输出目录已存在 | 使用新的 `--output`；随后重新划分并使用对应的新配置 |
| `Policy/dataset family mismatch` | 核对采集地图/预设与划分策略，补齐或移除对应家族 |
| `Every source track must contain exactly the configured seeds` | 统一采集 `augmentation.seeds` 与策略 `expected_seeds`；不要混用单 seed `main.py collect` 的目录 |
| 缺少 metadata、坐标契约不符 | 重新用本教程入口生成完整 ego 数据；不要通过 legacy 参数掩盖错误 |
| SHA-256 不匹配 | 核对是否改过帧内容、换过数据或清单；恢复原始实验数据，或为新数据重新划分和训练 |
| CUDA 不可用 / 显存不足 | 核对驱动与 wheel，降低 `--batch-size`，必要时使用 CPU |
| CUDA 确定性运算报错 | 在启动进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 训练中没有 `.pth` / 中断后不能续训 | 当前仅在完整训练结束时保存权重，且没有恢复训练接口；参见第 6 节 |
| 验证高但实际效果差 | 排查跨家族泄漏、坐标/侧别语义、传感器噪声差异和虚警处理，再补充实车与下游验证 |

## 10. 保存实验与协作注意事项

每次正式实验至少保留：两个仓库的 commit 与本地改动、采集配置和地图/预设、完整数据及 metadata、
划分策略与三分区清单、实际训练配置、best/last 权重、训练 split、metrics 和最终测试报告。
checkpoint 会记录 resolved config 与部分运行环境，但不能替代对数据和全部依赖的归档。

可在已创建的实验目录中补充环境记录：

```bash
.venv/bin/python -m pip freeze > runs/dgcnn_mixed_run01/requirements.freeze.txt
git rev-parse HEAD > runs/dgcnn_mixed_run01/sidenet_commit.txt
git -C ../bitfsd-generator rev-parse HEAD > runs/dgcnn_mixed_run01/generator_commit.txt
git diff > runs/dgcnn_mixed_run01/sidenet_changes.patch
git -C ../bitfsd-generator diff > runs/dgcnn_mixed_run01/generator_changes.patch
cp configs/dgcnn_mixed.yaml configs/mixed_split.yaml splits/mixed_ego.json runs/dgcnn_mixed_run01/
cp ../bitfsd-generator/config/perceive_mixed.yaml runs/dgcnn_mixed_run01/
```

`git diff` 只记录未暂存的已跟踪文件改动；暂存改动需另存 `git diff --cached`，新增未跟踪文件也需单独归档。
配置、代码和 `splits/` 适合提交 Git；被忽略的数据与权重应通过实验存储、Release 附件或其他共享位置分发，并附 SHA-256。
在新的实验目录复制清单尤其重要，避免下一次准备数据覆盖 `splits/mixed_ego.json` 后丢失实验对应关系。

同一数据集比较 DGCNN、PointNet、PointNet++、Transformer 时，应复制本次生成的训练配置，
保留相同数据与划分字段，再参考对应模型配置调整 `model` 和独立的保存目录。
不要直接切换到另一份默认配置后将不同数据、不同 fold 的结果横向比较。
增加观测 seed 不能替代增加赛道几何多样性；二分类 SideNet 也不会自动去除 detector 虚警。

进一步阅读：[混合数据集设计与历史统计](MIXED_DATASET.md)、[工程说明](ENGINEERING_REVIEW.md)、
[SideNet README](../README.md)。`TRAINING_READINESS.md` 是特定机器的历史检查记录，不是新环境安装前提。
