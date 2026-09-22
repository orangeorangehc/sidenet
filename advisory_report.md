# FSD 锥桶左右分类项目汇报

> **历史快照说明（2026-08-20）：**本文件保留当时的阶段汇报，不再作为当前运行说明。历史 DGCNN 95.6% 已在固定 legacy 条件下完整复现到 94.68%（4113/4344），能力量级得到确认，但 random-frame 条件不能代表 unseen-track 泛化。当前系统定位、ego-frame / track-family holdout 协议和下一步见 [`docs/ENGINEERING_REVIEW.md`](docs/ENGINEERING_REVIEW.md) 与 [`README.md`](README.md)。

## 问题定义

Formula Student Driverless（FSD）比赛中，赛车需要通过锥桶识别赛道边界。规则规定：
- 蓝色锥桶标记左侧边界（Cone_Left）
- 黄色锥桶标记右侧边界（Cone_Right）
- 赛道闭环 200-500m，锥桶间距 5-15m，弯道处加密

现有流程：PointPillars 等 3D 检测器输出「这里有锥桶」，但**不做左右分类**（输出 label 统一为 Cone）。左右分类要么靠手工规则（离中线左边=Left），要么直接不做。我们提出了一个轻量级的神经网络 SideNet，挂在检测器后面，对每个检测到的锥桶判断 Left/Right。

## 已完成工作

### 1. 数据平台：bitfsd-generator
- 根据 FSD 赛道规则生成合成赛道（直道+弯道分段，曲率自适应锥桶间距）
- 加载真实 FSD 赛道（FSE22/23/24, FSG19/21/23/24, FSCZ24 等 15 条真实赛道）
- 感知仿真管线：采样 ego 位姿 → LiDAR 视场过滤 → 噪声注入（dropout/抖动/虚警）
- 输出 OpenPCDet 兼容格式，可直接用于训练

### 2. benchmark 实验（单赛道内随机 80/20 split，714 帧，~22000 锥桶）

| 方法 | 验证准确率 | 参数量 | 核心思路 |
|------|-----------|--------|---------|
| k-NN (k=5) | 47.3% | 0 | 纯几何投票，等同于瞎猜 |
| PointNet | 79.9% | 1.1M | 全局 max pool |
| PointNet++ | 83.1% | 65K | 层次采样 + 插值 |
| Transformer | 92.4% | 100K | 自注意力 |
| **DGCNN** | **95.6%** | 91K | EdgeConv（k-NN 边特征） |

### 3. 关键发现
- k-NN 47% 说明**空间近邻直接投票无用**——左右靠的不是谁离得近
- EdgeConv 显式编码 `（邻居坐标 - 自己坐标）` 捕捉了局部几何，显著优于 Transformer 的隐式注意力
- PointNet++ 采样丢信息，在小点集（~30 点/帧）上不如扁平结构
- DGCNN 91K 参数、推理 <1ms/帧，适合部署

## 当前瓶颈

1. **Split 方式不可靠**：当前是随机 80/20 分帧，同一赛道帧同时在 train 和 val。需改为按赛道 split（hold-out 2-3 条赛道测泛化）
2. **缺乏错误分析**：剩下的 4.4% 错在哪里？弯道、稀疏区域还是虚警？
3. **无端到端验证**：未集成到 PointPillars 后测完整 pipeline
4. **数据集不公开**：无法让其他人复现

## 拟投稿方向（征求导师意见）

| 方向 | 核心贡献 | 风险 | 适合 |
|------|---------|------|------|
| **A. 新方法** | 设计 DGCNN+TrackPrior，融入赛道先验 | 高 | 有创新，做出来可冲好会 |
| **B. 新 Benchmark** | 释放标准 FSD 锥桶分类数据集 + 系统 benchmark | 低 | RAL/IROS short paper |
| **C. 系统部署** | 感知→左右分类端到端，Jetson 上跑 | 中 | 偏工程报告 |
| **D. 小样本泛化** | 新赛道只需标 10% 锥桶 | 高 | 有 novelty，周期长 |

**目前倾向方向 B**：问题定义清晰、方法对比完整、代价值得、可复现。目标投 RAL short paper 或 ICRA 短文。

## 建议讨论的问题

1. 方向选择：B 是否可行？有没有结合 A/C 的可能？
2. 是否需要补充真实锥桶标注数据（非合成 percep）？
3. contribution 够不够 RAL？要不要先投 workshop 试水？
4. 是否需要在实车上验证？
