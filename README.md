# SideNet

SideNet 是 FSD 锥桶检测后的边界侧别分类器：输入一帧检测锥桶集合，输出与每个输入锥桶一一对应的 `Left/Right` 概率。当前推荐 baseline 是 DGCNN；PointNet、PointNet++、Transformer 和两个非学习 baseline 用于对照。

工程设计与证据审计见 [`docs/ENGINEERING_REVIEW.md`](docs/ENGINEERING_REVIEW.md)。

首次使用请按 [数据生成、三分区划分与训练操作指南](docs/TRAINING_GUIDE.md) 执行，包含从克隆安装到独立测试的完整命令。

独立配置训练环境（Linux / WSL）：

```bash
bash setup_env.sh                  # 自动检测 CPU / NVIDIA GPU，配置本项目 .venv
bash setup_env.sh --check          # 只检测，不安装或更新依赖
bash setup_env.sh --device cpu     # 指定 CPU 版 PyTorch
bash setup_env.sh --device cuda    # 要求可用的 CUDA 环境
```

脚本支持 Python 3.12/3.13；有 `uv` 时可以自动下载 Python，否则使用本机 Python + venv/pip。
默认复用 `.venv`，安装 `requirements.txt` 和 PyTorch 2.8.0，并检查实际张量前向/反向运算。
自动选择驱动支持的 `cu128` 或 `cu126`；未检测到支持的 GPU 时选择 CPU。
已有可用的 cu126/cu128 PyTorch 会优先保留相应 CUDA 版本。可用 `--cuda cu126` 显式选择，
用 `--venv .venv-new --python 3.12` 建立新环境。脚本不安装系统驱动，也不启动训练。
生成器使用它自己的 `../bitfsd-generator/setup_env.sh`，两个脚本不依赖对方仓库。

生成、划分和训练现已提供三个独立脚本，参数分别放在专用 YAML 中：

```bash
bash ../bitfsd-generator/generate_data.sh  # config/generate_data.yaml
bash split_data.sh                       # configs/split_data.yaml
bash start_training.sh                   # configs/train.yaml
```

默认输出使用新的 `sidenet_data_pipeline_ego` 数据目录。训练参数与自动生成的划分配置分开保存，
重新划分不会覆盖调参。三个脚本均支持 `--config` 和 `--dry-run`。
完整参数、路径和产物说明见 [三个脚本的使用指南](docs/WORKFLOW_SCRIPTS.md)。

## 当前结论

- 默认任务语义是 **track-global directed side**：Left/Right 相对赛道规定的 canonical driving direction，而不是任意瞬时 ego heading。
- 默认输入是 **ego/LiDAR frame**，采用 `+x forward, +y left`。bitfsd-generator 会执行
  \[
  p_i^E=R(-\psi_E)(p_i^W-t_E)
  \]
  并在每个 track 的 `metadata.yaml` 中记录契约。
- 默认评测是 **track-family holdout**，固定 seed、保存 split manifest，并报告 per-track/worst-track 指标。
- 历史 95.6% 已按 legacy world-coordinate + random-frame 条件完整复现到 94.68%（4113/4344）；能力量级成立，但不能当作 unseen-track 或闭环结论。
- 严格 SE(2)/E(2)-equivariant network 暂不作为默认下一步；先验证 ego-frame DGCNN 的 grouped generalization 和 downstream boundary KPI。

## 系统边界

```text
LiDAR → PointPillars / CenterPoint → detected cones in LiDAR frame
      → SideNet probabilities → boundary association
      → centerline/path → controller
```

SideNet 不负责检测，也不凭空消除 false positive。binary 模式下 FP 必须由 detector threshold、SideNet abstention 或 boundary association 拒绝。训练文件默认排除 FP；generator 可显式导出 `Cone_Unknown`，但这要求把 `data.num_classes` 和系统接口一起升级为三分类。

## 数据生成

已经准备好的虚拟 + 真实赛道多 seed 数据、家族划分与训练命令见
[`docs/MIXED_DATASET.md`](docs/MIXED_DATASET.md)。对应训练配置为 `configs/dgcnn_mixed.yaml`。

先从相邻的 bitfsd-generator 生成新的 manifest-backed ego-frame 数据：

```bash
cd ../bitfsd-generator
.venv/bin/python main.py collect --config config/perceive.yaml
```

默认输出到：

```text
bitfsd-generator/output/sidenet_data_ego/
├── dataset_manifest.yaml
├── FSE22/
│   ├── metadata.yaml
│   ├── cloud_0.txt
│   └── ...
└── ...
```

每个 `cloud_N.txt` 的 canonical v1 格式为：

```text
x y z dx dy dz heading score class_name
```

其中 `class_name` 为 `Cone_Left` 或 `Cone_Right`。generator 合成 score 目前仍为常数，因此推荐配置使用 `input_mode: xyz`；只有用真实 detector predictions 训练时才建议启用 `xyzs`。

`dataset_manifest.yaml` 是数据源真相。loader 不再扫描并静默纳入 output 下的遗留目录；缺失 metadata 时默认 fail closed。旧数据只能通过明确的 legacy config 使用。

### Side 语义和 flip

- `side_semantics: track_global`：逆向行驶时 boundary identity 不交换。
- `side_semantics: ego_relative`：逆向行驶时 Left/Right 交换。
- 默认 collection 不生成 reverse `_flip` 数据。
- 训练时的 Y reflection 表示对有向 ego-frame 几何做镜像，因此同步执行 `Left ↔ Right` label permutation。

## 训练

当前工作区的环境检查、独立测试集划分和可直接执行的训练命令见
[`docs/TRAINING_READINESS.md`](docs/TRAINING_READINESS.md)。该划分使用
`configs/dgcnn_current_ego.yaml`，将独立测试赛道排除在训练和验证之外。

```bash
# 推荐：ego-frame DGCNN + track-family holdout
python train_sidenet.py configs/dgcnn.yaml

# 其他模型和 baseline
python train_sidenet.py configs/pointnet.yaml
python train_sidenet.py configs/pointnet2.yaml
python train_sidenet.py configs/transformer.yaml
python train_sidenet.py configs/knn.yaml
python train_sidenet.py configs/lateral.yaml
```

命令行覆盖会更新真实的 nested config：

```bash
python train_sidenet.py configs/dgcnn.yaml \
  --epochs 50 --lr 0.0005 --batch-size 8 --seed 7 --device cuda
```

`--seed` 只改变 model/augmentation seed，保持 fold 不变；只有显式 `--split-seed` 才改变自动生成的 split。配置已给出 `val_groups` 时，fold 不随二者变化。

默认 holdout groups 为 `FSCZ24 / FSG24 / FSS22`。`FSE22_test` 与 `FSE22`、`FSS22_V1/V2`、`_flip` 和同源 synthetic seed variants 会自动归入同一个 track family，避免跨 split。

每次训练保存：

```text
runs/<architecture>/
├── <architecture>.pth          # selection metric 最优 epoch
├── <architecture>.last.pth     # 最后一 epoch
├── split_manifest.json         # frame ids、track groups、dataset SHA-256
└── metrics.jsonl               # micro/balanced/F1/calibration/macro/worst-track
```

checkpoint 还记录 SideNet Git commit、dirty-worktree 标记、Python/PyTorch 版本和完整 resolved config；generator manifest 同样记录 generator commit/dirty 状态。

训练不再把不同长度的 frame padding 后送入 DGCNN/PointNet/PointNet++；一个 optimizer batch 仍可累积多个 frame，但每个 set 独立 forward，避免 padding 污染 k-NN/global feature。

### 历史 random-frame 条件

最小 E0→E2 对照可直接使用三份固定配置：

```bash
# E0: historical world coordinates + deterministic random-frame smoke
python train_sidenet.py configs/dgcnn_legacy_random.yaml

# E1: same world corpus + track-family holdout
python train_sidenet.py configs/dgcnn_world_track_holdout.yaml

# E2: regenerated ego corpus + the same track-family holdout
python train_sidenet.py configs/dgcnn.yaml
```

该配置显式列出历史 15 个目录、声明 world frame 并固定 random split seed，只用于 E0 reproducibility smoke test，不代表泛化评测。

## 推理与 detector 接入

### 有 GT 的 manifest dataset

```bash
python infer.py \
  --ckpt runs/dgcnn/dgcnn.pth \
  --data-dir ../bitfsd-generator/output/sidenet_data_ego
```

v2 checkpoint 默认只评测其 `split_manifest` 记录的 validation frames，并校验 dataset SHA-256；需要诊断整个 corpus 时显式加 `--all-frames`。不要用 `--allow-dataset-mismatch` 生成正式结果。

### 无 GT 的 detector 文本

`--no-gt` 现在会保留行尾仅为 `Cone` 的检测结果，并读取倒数第二列 score：

```bash
python infer.py \
  --ckpt runs/dgcnn/dgcnn.pth \
  --data-dir /path/to/detector_frames \
  --no-gt --coordinate-frame ego --allow-legacy-data \
  --min-confidence 0.7 --output-dir runs/predictions
```

低于 `--min-confidence` 的结果输出为 `Unknown`，不会被强制计为某一侧。

Python adapter 可直接接 OpenPCDet `pred_boxes` / `pred_scores`：

```python
from infer import SideNetPredictor

predictor = SideNetPredictor("runs/dgcnn/dgcnn.pth")
side, confidence, probability = predictor.predict_detected_cones(
    pred_boxes[:, :3], pred_scores
)
```

adapter 会校验 checkpoint 的 coordinate contract；不能用命令行参数把 world-trained checkpoint 伪装成 ego-frame model。

## 模型

| 名称 | 当前角色 | 主要性质 |
|---|---|---|
| Lateral sign | ego-frame sanity baseline | `y>=0 → Left`；不学习 |
| train-reference k-NN | deployable point baseline | 从 train points 投票；不再读取同帧 GT neighbor label |
| PointNet | set-global baseline | per-point MLP + global max pool |
| PointNet++ | hierarchical baseline | deterministic FPS，支持小于 sampling count 的 frame |
| Transformer | global-context baseline | all-to-all self-attention |
| DGCNN | 推荐 baseline | dynamic k-NN + EdgeConv；不是严格 SE(2)-equivariant |

`input_mode: xy / xyz / xyzs` 现在会真正选择 2/3/4 个输入通道，所有模型均覆盖小点集回归测试。

## 历史结果：只作为项目过程记录

旧 README 报告过以下 random-frame 数字：

| 方法 | 报告 accuracy | 当前 artifact 状态 |
|---|---:|---|
| k-NN oracle heuristic | 47.3% | 旧实现已替换；不可部署 |
| PointNet | 79.9% | checkpoint metadata 约 79.85% |
| PointNet++ | 83.1% | checkpoint metadata 约 83.12% |
| Transformer | 92.4% | 现存 checkpoint metadata 约 91.25%，不一致 |
| DGCNN | 历史 95.6%；本次 94.68% | 已保存 best/last checkpoint、split、metrics 与独立 eval |

这些值来自 world-coordinate、近重复 observation 的 random frame split，模型间也未保存共同 split；不能与新 track-holdout 结果横向混用。新的实验结果应从 `metrics.jsonl + split_manifest.json + checkpoint` 联合发布。

## 测试

环境中若安装了不兼容的全局 pytest plugin，可禁用自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

测试覆盖 coordinate contract、manifest/stale-directory 行为、family split、reflection label permutation、CLI override、无 GT parser、全部 input mode、DGCNN 单点和 PointNet++ deterministic small-N。

## 项目结构

```text
SideNet/
├── sidenet.py                 # 模型与 model factory
├── sidenet_data.py            # 数据契约、loader、augmentation、split
├── train_sidenet.py           # 可复现训练/评测/checkpoint
├── infer.py                   # 评测、no-GT CLI、detector adapter
├── config.yaml                # 推荐 DGCNN 配置
├── configs/                   # 模型、baseline、legacy E0 配置
├── tests/                     # 回归测试
└── docs/
    ├── ENGINEERING_REVIEW.md
    └── backprop_and_differentiability.md
```

## English summary

SideNet classifies detected FSD cones into directed track-boundary sides. The recommended pipeline uses ego/LiDAR-frame coordinates, manifest-backed datasets, deterministic track-family holdouts, best/last checkpoints, and per-track tail metrics. A full legacy-condition run reproduced the historical capability scale at 94.68%, but this is not an unseen-track or closed-loop result. See the engineering review for the full system argument and experiment gates.

## License

[MIT](LICENSE) © Yuchen Fan
