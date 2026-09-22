# 反向传播与可导性：什么能训练，什么不能

> 以 CenterPoint + SideNet 为例，理解神经网络训练的边界。

## 什么是反向传播

反向传播是神经网络学习的核心机制：**损失函数对每个参数的偏导数，沿计算图反向传递，告诉优化器"这个参数往哪个方向调、调多大"**。

```
Forward:  输入 → Layer1 → Layer2 → ... → 输出 → Loss
                                              ↑
Backward:  grad ← ∂Loss/∂LayerN ← ... ← ∂Loss/∂Layer1
                    ↑ 链式法则逐层传递
```

每一个中间操作必须满足：给定输出梯度，能算出输入梯度。这就是**可导**。

## 可导操作：训练的灵魂

```python
# ✅ 这些都是可导的
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2           # 乘法: ∂y/∂x = 2，反向传 dy * 2
y = x.sin()         # sin: ∂y/∂x = cos(x)
y = x @ W           # 矩阵乘: ∂y/∂x = W^T
y = F.softmax(x)    # softmax: 有解析梯度
y = F.cross_entropy(logits, labels)  # 内部 softmax + log + NLL，全程可导
```

训练需要的是框架能够为当前计算路径定义有用的梯度。操作不必处处平滑：例如 ReLU 在 0 处不可微，但框架采用约定的次梯度；真正的问题通常是离散 index/selection 无法对“选中了谁”提供连续梯度。

## 不可导操作：训练的断点

```python
# ❌ 不可导 / 梯度为 0
x = torch.tensor([1.5, -2.3, 0.8], requires_grad=True)

y = x.argmax()       # 输出: 0 (索引)，离散变化，梯度为 0 或不存在
y = x > 0            # 输出: [True, False, True]，布尔值不可导
y = sorted_x = x.sort()  # values 可回传；离散 permutation/index 不可导
y = x[0] if x[0] > 0 else x[1]  # 条件分支，不同输入走不同路径
```

这些操作在推理时无处不在（选最大值的索引、NMS 过滤、阈值判断），但在训练时**必须绕过它们**。

## 完整案例：CenterPoint 的"分身术"

CenterPoint 用同一个 heatmap 输出，走两条不同路径完成训练和推理：

```
CenterHead 输出: (B, C, H, W) 密集 heatmap，每个像素都有预测值
                          │
        ┌─────────────────┼─────────────────┐
        │ 训练路径 (可导)                      │ 推理路径 (不可导)
        │                                     │
        ▼                                     ▼
┌───────────────────┐                 ┌──────────────────┐
│ Target Assigner   │                 │ sigmoid          │
│ GT 框 → heatmap 对应位置            │ heatmap → 概率图  │
│ → Focal Loss (分类)                │                  │
│ → L1 Loss (回归)                   ▼                  │
│                                     │
│ ✅ 只在 GT 位置算 loss              │ argmax           │
│ ✅ 连续值，可导                     │ 每个类别找最高响应 │
│                                     │ → 离散像素坐标    │
└──────────┬────────────┘             │ ❌ 不可导         │
           │                          │                  │
           ▼                          ▼                  │
    backward() 反向传播          ┌──────────────┐       │
    梯度 → CenterHead            │ decode 坐标   │       │
    梯度 → Backbone              │ heatmap 像素 → │       │
    梯度 → VFE                   │ 世界坐标       │       │
                                 │ ❌ 不可导      │       │
                                 └──────┬───────┘       │
                                        │               │
                                        ▼               │
                                 ┌──────────────┐       │
                                 │ NMS 去重      │       │
                                 │ IoU 计算      │       │
                                 │ 阈值过滤      │       │
                                 │ ❌ 全都不可导  │       │
                                 └──────┬───────┘       │
                                        │               │
                                        ▼               │
                                 检测框 (x,y,z,...)     │
```

### 关键洞察

**同一个网络、同一组权重，训练走可导的"特殊通道"，推理走不可导的"后处理通道"**。训练时的 loss 直接作用在 heatmap 连续值上，跳过了 argmax/NMS 等不可导操作。

这就好比学投篮：训练时教练直接告诉你"手腕角度偏了 5 度"（连续梯度信号），不需要你先投中再评价。

## 当前 coordinate-only SideNet：为什么采用串联

```
CenterPoint 推理 → 检测框 (离散坐标)
                        │
                        ▼
                  SideNet → Left/Right
```

当前 SideNet 在 Postprocessor 之后接收离散选择出的 box。按这个接口，side loss 不能穿过 argmax/NMS 去改变“哪些 box 被选中”，所以当前 DGCNN baseline 采用独立训练。

```
✅ SideNet 单独训练:
   GT 坐标 → SideNet → CrossEntropyLoss(Left/Right)

❌ 当前 post-NMS coordinate 接口不能直接联合训练:
   PointCloud → CenterPoint → argmax (断点!) → SideNet → loss
```

这不是说整个系统在数学上只能两阶段。若改变训练接口，可以在 GT center 或 dense BEV feature 上加 auxiliary side head，或把 Left/Right 设为 detector 的 multi-task target；这些路径不需要对 NMS 的离散选择求导。准确结论是：**当前 post-NMS 坐标接口是两阶段；端到端需要改训练目标或接口。**

## 什么可以训练，什么不行

| 操作 | 可导？ | 常见于 |
|------|--------|--------|
| Linear / Conv / 矩阵乘 | ✅ | 所有神经网络 |
| sin/cos/exp/log | ✅ | 激活函数、位置编码 |
| softmax / sigmoid | ✅ | 概率输出 |
| ReLU / GELU | ✅ | 激活函数 |
| LayerNorm / BatchNorm | ✅ | 归一化 |
| 加法 / 点乘 | ✅ | 残差连接 |
| `x[a:b]` 固定切片 | ✅ | token 拼接 |
| `torch.where(condition)` | 部分✅ | 注意：只对选择的分支求导 |
| **argmax / argmin** | ❌ | heatmap 解码 |
| **阈值判断 (>, <)** | ❌ | NMS 过滤 |
| **排序 / Top-K** | 部分✅ | selected values 可回传；离散 index/排列不可导 |
| **稀疏索引 gather** | 部分✅ | 对 gathered source values 可导；对 index 不可导 |
| **条件分支 (if)** | ❌ | 控制流中断计算图 |
| **离散采样** | ❌ | Gumbel-Softmax 可作为近似 |

## 记忆口诀

> 训练看 loss graph，推理看后处理。  
> 连续值都行，离散值都断。  
> argmax 解码，NMS 过滤，这两步只能在推理路径上。  
> 要让 side loss 改变 detector 的离散选择，需要重写训练接口；也可以在 NMS 前增加独立可导的 auxiliary target。
