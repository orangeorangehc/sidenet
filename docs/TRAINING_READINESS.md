# 当前数据的训练准备情况（2026-09-22）

本页记录最初的单 seed 真实赛道基线。后续生成的虚拟 + 真实多 seed 混合数据和新划分见
[MIXED_DATASET.md](MIXED_DATASET.md)，使用 `configs/dgcnn_mixed.yaml` 启动。

当前环境可以开始 DGCNN 训练。使用 SideNet 自己的 `.venv/bin/python`；系统
`python3` 与这个环境不同。已验证一个完整 epoch、验证集评估、best/last checkpoint
保存和重新加载。没有启动 300 epoch 正式训练，也没有对独立测试集评估模型。

## 环境与数据

- Python 3.12.14，PyTorch 2.8.0+cu128（CUDA runtime 12.8）。
- NumPy 2.5.3，PyYAML 6.0.3。
- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存，驱动 591.86。
- 检查时内存约 5.6 GiB 可用、工作盘约 934 GiB 可用。
- Codex 沙箱内 GPU 访问受限；已在获准的沙箱外实际完成 CUDA 训练。
  在本机终端使用下面命令即可，不要依据沙箱中的 `cuda.is_available() == False`
  判断本机没有 CUDA。

数据根目录为 `../bitfsd-generator/output/sidenet_data_ego`（相对于 SideNet）。
这是基于真实赛道地图生成的感知仿真数据，目前没有纯虚拟赛道数据。

数据有 14 个赛道目录、12 个赛道家族、643 个非空帧、21,154 个锥桶观测。
Left 为 10,557 个，Right 为 10,597 个，每帧 7–83 个锥桶。
SideNet loader 已成功读取全部数据；9 列格式、左右标签、有限数值和坐标契约检查均通过。
当前数量足够完成基线训练，但同一赛道的相邻帧相关性较高，不能把观测数当成独立赛道数，
也不能据此保证实车效果。

## 固定划分

划分按赛道家族进行，不复制或移动原始数据。

| 分区 | 赛道家族 | 帧数 | 锥桶观测数 | Left / Right |
|---|---|---:|---:|---:|
| 训练 | FSE22、FSE23、FSE24、FSG19、FSG23、FSI24、FSO20、FSS19 | 430 | 14,689 | 7,293 / 7,396 |
| 验证 | FSCZ24、FSG21 | 87 | 2,723 | 1,404 / 1,319 |
| 独立测试 | FSG24、FSS22 | 126 | 3,742 | 1,860 / 1,882 |

帧数比例约为 67% / 13% / 20%。三者的赛道家族交集为零，每一帧恰好属于一个分区。
`FSE22_test` 名字中的 `_test` 不决定用途，它与 FSE22 同属训练集；
`FSS22_V1` 和 `FSS22_V2` 一起留作独立测试。

[划分清单](../splits/current_ego.json) 保存全部帧 ID、分组、数量和 SHA-256。
[专用训练配置](../configs/dgcnn_current_ego.yaml) 的 `include_dirs` 只包含训练和验证目录，
再通过 `val_groups` 选择验证集。训练程序实际生成的 split 已与该清单逐帧比对一致。
清单用于审计；现有训练入口仍通过配置实时划分。如果重新生成或替换数据，需要重新核对清单。

验证集参与 best checkpoint 选择；独立测试集只用于训练完成后的最终评估。
现有 `infer.py` 默认评估 checkpoint 的验证集，不是这里预留的独立测试集。
其 `--all-frames` 也不会自动绕过 checkpoint 的 `include_dirs` 去加载测试赛道。

## 启动正式训练

```bash
cd /home/hc/lidar-cone-perception/SideNet

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u train_sidenet.py configs/dgcnn_current_ego.yaml \
  --device cuda
```

配置使用 DGCNN、xyz 输入、batch size 16、300 epochs；CUDA workspace 设置用于满足
已启用的确定性运算要求。单轮试跑的训练和验证耗时约 2.9 秒，按此线性估算 300 轮约
15 分钟，实际耗时受 GPU 负载等因素影响。试跑证明流程可执行，不代表模型已经训练充分。

输出目录：

```text
runs/dgcnn_current_ego/
├── dgcnn.pth
├── dgcnn.last.pth
├── split_manifest.json
└── metrics.jsonl
```

当前训练器在全部 epochs 完成后写入 best/last checkpoint；metrics 每轮写入。
首次试跑产物位于 `/tmp/sidenet-preflight-d5RxCi/`，正式训练不会覆盖它。
仓库尚无初始提交，试跑出现的 Git HEAD 提示不影响训练成功，但 checkpoint 暂无 commit ID。

训练完成后，可先复核验证集结果：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python infer.py --ckpt runs/dgcnn_current_ego/dgcnn.pth \
  --data-dir ../bitfsd-generator/output/sidenet_data_ego --device cuda
```
