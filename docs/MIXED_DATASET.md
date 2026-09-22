# 虚拟赛道与真实赛道的多 seed 数据集

数据已生成到 `bitfsd-generator/output/sidenet_data_mixed_ego/`。
使用观测 seed **42、43、44**，包含 54 个目录、2,535 帧、73,394 个锥桶观测。
其中真实赛道部分也是地图上的感知仿真，并非实车 LiDAR 回放。

## 过拟合与评估

混合这两种来源不会必然导致过拟合；不同 seed 的定位噪声、漏检和检测位置扰动
可以增加观测变化。但当前 seed 不改变赛道几何，本数据集仍然只有 12 个真实赛道家族
和 4 个虚拟预设，不能把多 seed 当作新增独立赛道。

本次通过以下安排减少数据泄漏和偏置：

- 同一家族的全部 seed、`_test`、`_V1/V2` 版本保持在同一个分区。
- 相似的 `simple_oval`、`tight_oval`、`hairpin` 全部用于训练，`triangle` 用于验证。
  这四个预设的中心线已检查，没有发现非相邻线段的穿越交叉；这不是完整赛规认证。
- 沿用此前真实赛道的家族划分，最终测试仍保留 FSG24、FSS22。
- 每个来源只生成三个 seed，虚拟数据约占训练帧的 23.5%、训练锥桶观测的 13.9%，
  避免大量重复虚拟布局占据训练数据主体。
- 验证时同时查看真实赛道 FSCZ24、FSG21 的 per-track 指标。
  简单虚拟赛道可能明显容易，混合总体 accuracy 不能代替真实赛道泛化表现。

是否过拟合以及影响多大，需要正式训练后比较训练与验证指标的走势；一次短训练无法判定。
应在验证表现最好的 epoch 选择模型，若训练继续改善而真实验证停滞或下降，则考虑减少
训练轮次、增加几何多样性或调整正则化。测试集只在方案确定后用于最终评估。

## 划分结果

| 分区 | 虚拟帧 | 真实帧 | 总帧数 | 锥桶观测数 |
|---|---:|---:|---:|---:|
| 训练 | 396 | 1,290 | 1,686 | 51,075 |
| 验证 | 210 | 261 | 471 | 11,087 |
| 独立测试 | 0 | 378 | 378 | 11,232 |

每条源赛道的三个 seed 版本都计入上述帧数。

| 来源 | 训练家族 | 验证家族 | 测试家族 |
|---|---|---|---|
| 虚拟 | simple_oval、tight_oval、hairpin | triangle | 无 |
| 真实 | FSE22、FSE23、FSE24、FSG19、FSG23、FSI24、FSO20、FSS19 | FSCZ24、FSG21 | FSG24、FSS22 |

`FSE22_test_s42/s43/s44` 与 FSE22 一起训练；
`FSS22_V1_s42/s43/s44`、`FSS22_V2_s42/s43/s44` 全部用于测试。
训练/验证/测试家族交集为零。

- [划分策略](../configs/mixed_split.yaml)：在训练前指定家族归属和 seed。
- [完整划分清单](../splits/mixed_ego.json)：每帧 ID、来源统计、数据 SHA-256。
- [训练配置](../configs/dgcnn_mixed.yaml)：只加载训练和验证目录，测试目录已排除。

## 生成与复现

采集入口为 generator 的 `collect_multiseed.py`，复用现有 `collect` 流程，
为真实赛道也添加 `_s42` 等后缀，并在 metadata 中保存 `source.type/track/seed`。
一个根 manifest 列出所有 seed 的数据；只有全部成功后才将数据集写入目标目录。

以下是本次生成和划分命令。数据已经存在，无需再次生成。
采集脚本会拒绝覆盖已有目录；如要新建版本，使用 `--output` 指定新目录，
并在划分时通过 `--data-dir` 指向相应目录。

```bash
cd /home/hc/lidar-cone-perception/bitfsd-generator
../SideNet/.venv/bin/python collect_multiseed.py --config config/perceive_mixed.yaml

cd ../SideNet
.venv/bin/python prepare_mixed_split.py
```

使用已验证的 SideNet 虚拟环境即可完成生成，无须额外安装依赖。
配置、源地图、预设、代码版本及处理顺序相同，才能复现观测结果。
`prepare_mixed_split.py` 校验每条来源的 seed 是否齐全，检查数据契约和家族隔离，
并生成训练配置与划分清单。更换数据后应重新运行此脚本。

## 开始训练

```bash
cd /home/hc/lidar-cone-perception/SideNet
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u train_sidenet.py configs/dgcnn_mixed.yaml --device cuda
```

默认 300 epochs、batch size 16，保存到 `runs/dgcnn_mixed/`。
新数据集已在 CPU 上完成一个 epoch 的训练、验证、best/last 保存检查，约 5.5 秒。
测试集没有参与试跑。正式 GPU 训练时间需按新数据规模重新观察。
模型保存在全部 epochs 结束时，metrics 每轮写入。

目前 `infer.py` 默认读取 checkpoint 的验证集；`--all-frames` 也不会自动纳入
被 `include_dirs` 排除的测试目录。最终独立测试应使用划分清单中的 test 帧和对应 SHA-256
单独加载评估，不能把默认验证结果当作测试结果。

## 校验记录与已知限制

- 全部数据可被 SideNet loader 读取；坐标有限、标签合法、每行 9 列。
- 实际训练产生的 split 与保存清单一致，训练及验证中使用的测试帧数为零。
- 本次相关测试 19 项通过，包括多 seed 导出、覆盖保护、组合后缀分组及防跨集回归。
- 完整数据契约测试另外有两项既有失败：`test_world_to_ego_canonicalization`
  调用 generator 不存在的 `world_to_ego_xyz`（当前实现名为 `_world_to_ego`）；
  `test_reverse_driving_side_semantics_are_explicit` 调用尚未实现的 `side_semantics`
  参数。本次使用已有导出接口、固定 `track_global` 且关闭反向采集，已实测可训练。
- 仓库尚无初始提交，训练结束时的 Git HEAD 提示不会阻止保存，但 checkpoint 无 commit ID。
