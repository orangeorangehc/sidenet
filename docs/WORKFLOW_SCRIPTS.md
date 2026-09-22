# 用三个脚本完成生成、划分和训练

三个步骤分别读取自己的 YAML。修改参数后运行对应脚本即可；脚本不会安装环境。
请先在两个项目中分别运行 `bash setup_env.sh`。

| 步骤 | 脚本 | 手动编辑的配置 |
|---|---|---|
| 批量生成 | `bitfsd-generator/generate_data.sh` | `bitfsd-generator/config/generate_data.yaml` |
| 划分训练/验证/测试 | `SideNet/split_data.sh` | `SideNet/configs/split_data.yaml` |
| 开始训练 | `SideNet/start_training.sh` | `SideNet/configs/train.yaml` |

## 按顺序执行

```bash
cd ~/lidar-cone-perception/bitfsd-generator
bash generate_data.sh

cd ../SideNet  # 容器中如果目录名是小写，改成 cd ../sidenet
bash split_data.sh
bash start_training.sh
```

脚本根据自身位置定位项目，可以从其他目录通过绝对路径调用。
默认使用各自项目的 `.venv/bin/python`；如使用自定义环境，设置解释器的绝对路径：

```bash
PYTHON_BIN=/path/to/venv/bin/python bash start_training.sh
```

所有 YAML 中的相对路径均相对于**对应项目根目录**，不相对于 YAML 所在目录或终端当前目录。
配置也可以使用绝对路径，例如将生成数据放在容器的数据盘。
项目名称可以是 `SideNet` 或 `sidenet`；脚本不硬编码这个目录名。

## 1. 生成配置

编辑生成器中的 `config/generate_data.yaml`：

- `real_tracks`：真实地图文件，来自生成器 `data/`。
- `synthetic_tracks`：合成预设名，形状定义在 `config/track_presets.yaml`。
- `sensor`、`ego`、`perception`：LiDAR 范围、采样间距、定位噪声、漏检等。
- `augmentation.seeds`：观测种子，默认 `[42, 43, 44]`。
- `output.dir`：输出数据集目录，默认 `output/sidenet_data_pipeline_ego`。

当前多 seed 流程要求 `augmentation.flip: false` 和 `output.coordinate_frame: ego`。
默认 14 个真实地图 + 4 个合成预设，共产生 54 个赛道目录。seed 不改变赛道几何。
已有输出目录会报错；更换 `output.dir` 可生成新的数据集。

## 2. 划分配置

编辑 SideNet 中的 `configs/split_data.yaml`：

- `data_dir`：必须指向上一步的输出目录。
- `expected_seeds`：必须与生成配置的 `augmentation.seeds` 一致。
- `real`、`synthetic` 下的 `train`、`validation`、`test`：指定赛道家族归属。
- `output.data_config`：自动生成的数据配置，默认 `configs/train_data.yaml`。
- `output.manifest`：三分区帧清单，默认 `splits/pipeline_ego.json`。

同一家族的不同 seed、`_test`、`_V1/V2` 版本保留在同一分区。
当前流程要求三个分区均非空、均含左右两类，最终测试只包含真实地图。
它不按帧随机切分，也不移动或复制原始数据。

`train_data.yaml` 只保存训练/验证目录选择、坐标约定、划分信息和数据指纹。
不要手动编辑这个生成文件；改变数据或分区后重新运行 `bash split_data.sh`。
**重新划分不会覆盖 `configs/train.yaml` 中的训练参数。**
若已有按这套多 seed 流程生成的数据，可直接将 `data_dir` 指向它，从本步骤开始。

## 3. 训练配置

编辑 SideNet 中的 `configs/train.yaml`：

- `data_config`：引用上一步生成的 `configs/train_data.yaml`。
- `data`：输入特征和归一化；数据路径及分区由划分结果控制。
- `model`：模型类型、DGCNN 邻居数量和网络宽度。
- `training`：`epochs`、`batch_size`、`lr`、`weight_decay`、`device`、随机种子等。
- `augmentation`：训练时的旋转、缩放、反射和位置扰动。
- `save`：结果目录和模型文件名。
- `runtime`：OMP/MKL 线程数及确定性 CUDA 运算配置，在导入 PyTorch 前设置。

默认 DGCNN、300 epochs、batch size 16、`device: cuda`。
CPU 试跑可复制配置，修改 `training.device: cpu`、`training.epochs: 1`，
同时给 `save.save_dir` 设置独立目录，然后运行：

```bash
bash start_training.sh --config configs/train_smoke.yaml
```

训练脚本校验生成配置与三分区清单一致，测试目录不参与训练或验证。
训练器还会核对训练/验证数据指纹；更改数据文件后必须重新划分。
已有非空结果目录会报错，每次实验应修改 `save.save_dir`。

默认结果保存在 `runs/dgcnn_pipeline_run01/`：

```text
resolved_config.yaml  # 合并后实际传给训练器的配置
run_settings.yaml     # 本次手动编辑的训练配置副本
data_config.yaml      # 本次生成的数据配置副本
dataset_split.json    # 本次训练/验证/测试清单副本
split_manifest.json   # 训练器实际使用的训练/验证帧清单
metrics.jsonl         # 每轮指标
dgcnn.pth             # 验证指标最优的权重
dgcnn.last.pth        # 最后一轮的权重
```

沿用现有训练器：best/last 权重在全部 epochs 正常结束后保存，当前没有断点续训。
独立测试的 Python API 示例见 [训练指南第 8 节](TRAINING_GUIDE.md#8-方案冻结后评估独立测试集)。
使用本流程时，将其中的 `split_path` 改为本次结果目录的 `dataset_split.json`，
`ckpt_path` 改为对应的 `dgcnn.pth`；数据路径仍相对于 SideNet 项目根目录。

## 预览和多组实验

三个脚本都支持 `--config` 和 `--dry-run`：

```bash
# 在生成器目录；检查参数和赛道名，不生成数据
bash generate_data.sh --config config/generate_data.yaml --dry-run

# 在 SideNet 目录；数据生成后预览划分配置，不写划分结果
bash split_data.sh --config configs/split_data.yaml --dry-run

# 划分完成后预览最终训练参数，不启动训练
bash start_training.sh --config configs/train.yaml --dry-run
```

预览会检查配置及必要的上一步产物；不会代替正式划分的数据检查，
也不会验证 GPU 是否可用或运行模型。输出目录冲突同样会报错。

多组实验可复制三份 YAML，保持以下引用一致：

1. 生成配置的 `output.dir` 对应划分配置的 `data_dir`。
2. 生成配置的 `augmentation.seeds` 对应划分配置的 `expected_seeds`。
3. 划分配置的 `output.data_config` 对应训练配置的 `data_config`。
4. 每组实验使用自己的划分输出和 `save.save_dir`。

原来的 `collect_multiseed.py`、`prepare_mixed_split.py`、`train_sidenet.py` 命令仍然可用。
