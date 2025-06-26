## 📋 项目概述

本项目基于自主实现的深度学习框架 **MatrixSlow** 构建了一个完整的股票开盘价预测系统。该项目是深圳大学"基于Python的深度学习框架实现"的课程设计作品，完整实现了课程设计的所有要求。

### 核心特性

- ✅ **完全基于MatrixSlow**：所有神经网络组件均从零开始实现，无第三方框架依赖
- ✅ **多模型支持**：实现了LSTM、RNN、MLP以及带注意力机制的LSTM
- ✅ **完整流程**：包含数据预处理、模型训练、评估和可视化
- ✅ **深度分析**：提供模型对比、参数调优和性能分析
- ✅ **专业可视化**：生成高质量图表并支持自动保存

## 🎯 课程设计要求完成情况

| 要求 | 分值 | 完成情况 |
|------|------|----------|
| 基础LSTM实现与30天预测 | 40分 | ✅ 完成 |
| LSTM参数变化实验 | 10分 | ✅ 完成 |
| LSTM+注意力机制 | 15分 | ✅ 完成 |
| MLP和RNN对比分析 | 15分 | ✅ 完成 |
| **总计** | **80分** | **全部完成** |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.7
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
tqdm >= 4.62.0
```

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/Hunter0809/matrixslow-stock-prediction.git
cd matrixslow-stock-prediction

# 2. 安装依赖
pip install numpy pandas matplotlib tqdm
```

### 运行实验

```bash
# 运行完整实验（包含所有4个实验）
python main.py
```

## 📁 项目结构

```
matrixslow-stock-prediction/
│
├── core_framework.py          # MatrixSlow核心框架（计算图、节点基类）
├── operators.py               # 数学算子（加法、矩阵乘法、激活函数等）
├── neural_networks.py         # 神经网络实现（LSTM、RNN、MLP、注意力机制）
├── optimizers.py              # 优化器（SGD、Adam）
├── data_utils.py              # 数据处理工具
├── training_utils.py          # 训练工具函数
├── visualization.py           # 可视化和分析工具
├── main.py                    # 主程序入口
├── Daily_ZX.csv              # 中兴股票数据
└── README.md                  # 本文件
```

## 🔬 MatrixSlow框架原理

### 1. 计算图原理

MatrixSlow采用**动态计算图**机制，核心概念包括：

#### 1.1 计算图构建
```python
# 计算图由节点(Node)和边组成
# 节点表示变量或操作，边表示数据流向
class Graph:
    def __init__(self):
        self.nodes = []      # 所有节点列表
        self.node_dict = {}  # 节点名称索引
```

#### 1.2 自动微分原理

基于**链式法则**实现反向传播：

对于复合函数 $y = f(g(x))$，其导数为：
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

在计算图中，每个节点维护**雅可比矩阵**：
$$J = \frac{\partial \text{output}}{\partial \text{input}}$$

反向传播时，梯度通过雅可比矩阵传递：
```python
def backward(self, result):
    if self is result:
        self.jacobi = np.eye(self.dimension())
    else:
        self.jacobi = np.zeros((result.dimension(), self.dimension()))
        for child in self.children:
            if child.value is not None:
                self.jacobi += child.backward(result) * child.get_jacobi(self)
    return self.jacobi
```

### 2. 核心算子数学原理

#### 2.1 矩阵乘法 (MatMul)

前向传播：
$$C = A \times B$$

反向传播（雅可比矩阵）：
- 对A的导数：
  $$\frac{\partial C}{\partial A} = B^T$$
- 对B的导数：
  $$\frac{\partial C}{\partial B} = A^T$$

#### 2.2 激活函数

**Sigmoid函数**：
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
导数：
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Tanh函数**：
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
导数：
$$\tanh'(x) = 1 - \tanh^2(x)$$

**ReLU函数**：
$$\text{ReLU}(x) = \max(0, x)$$
导数：

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0 \end{cases}$$

## 🧠 神经网络模型原理

### 1. LSTM（长短期记忆网络）

LSTM通过门控机制解决梯度消失问题，核心公式如下：

#### 输入门 (Input Gate)
$$i_t = \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i)$$

#### 遗忘门 (Forget Gate)
$$f_t = \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f)$$

#### 候选记忆 (Candidate Memory)

$$\tilde{c}_t = \tanh(W_{ig}x_t + W_{hg}h_{t-1} + b_g)$$

#### 输出门 (Output Gate)
$$o_t = \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o)$$

#### 细胞状态更新
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

#### 隐藏状态更新
$$h_t = o_t \odot \tanh(c_t)$$

其中：
- $\sigma$ 表示Sigmoid函数
- $\odot$ 表示元素级乘法
- $W$ 表示权重矩阵，$b$ 表示偏置向量

### 2. RNN（循环神经网络）

RNN的核心公式相对简单：

$$h_t = \tanh(W_{ih}x_t + W_{hh}h_{t-1} + b_h)$$

输出：
$$y_t = W_{ho}h_t + b_o$$

### 3. MLP（多层感知机）

前向传播：
$$h^{(l)} = \sigma(W^{(l)}h^{(l-1)} + b^{(l)})$$

其中 $l$ 表示层数，$\sigma$ 为激活函数。

### 4. 注意力机制

本项目实现的注意力机制基于**加性注意力**：

#### 注意力分数计算
$$e_t = v_a^T \tanh(W_a h_t)$$

其中 $h_t$ 是第 $t$ 个时间步的隐藏状态。

#### Softmax归一化
$$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$

#### 上下文向量
$$c = \sum_{t=1}^{T} \alpha_t h_t$$

最终预测：
$$y = \text{Linear}(c)$$

## 📊 优化算法原理

### 1. SGD（随机梯度下降）

参数更新公式：
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

其中：
- $\theta$ 表示参数
- $\eta$ 表示学习率
- $\nabla_\theta L$ 表示损失函数对参数的梯度

### 2. Adam优化器

Adam结合了动量和自适应学习率：

#### 一阶矩估计（动量）
$$v_t = \beta_1 v_{t-1} + (1-\beta_1)g_t$$

#### 二阶矩估计（梯度平方）
$$s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2$$

#### 偏差校正
$$\hat{v}_t = \frac{v_t}{1-\beta_1^t}$$
$$\hat{s}_t = \frac{s_t}{1-\beta_2^t}$$

#### 参数更新
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

默认超参数：
- $\beta_1 = 0.9$（一阶矩衰减率）
- $\beta_2 = 0.999$（二阶矩衰减率）
- $\epsilon = 10^{-8}$（数值稳定性）

## 📈 损失函数

### MSE（均方误差）

$$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

梯度：
$$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$$

## 🔧 核心模块详解

### 1. MatrixSlow框架核心 (core_framework.py)

```python
class Node:
    def forward(self):
        """前向传播：递归计算父节点值，然后计算当前节点"""
        for parent in self.parents:
            if parent.value is None:
                parent.forward()
        self.compute()
    
    def backward(self, result):
        """反向传播：使用链式法则计算雅可比矩阵"""
        # 实现见上文自动微分原理
```

### 2. 参数初始化策略

#### He初始化（用于ReLU）
$$W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$$

#### Xavier初始化（用于Sigmoid/Tanh）
$$W \sim \mathcal{U}(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}})$$

### 3. 数值稳定性处理

1. **梯度裁剪**：防止梯度爆炸
   ```python
   if gradient_norm > max_norm:
       gradient = gradient * max_norm / gradient_norm
   ```

2. **Sigmoid数值稳定**：
   ```python
   x = np.clip(x, -500, 500)  # 防止数值溢出
   ```

3. **遗忘门偏置初始化**：
   ```python
   self.b_f = Parameter(np.ones((1, hidden_size)))  # 初始化为1
   ```

## 📊 实验设计原理

### 1. 时间序列预测

使用**滑动窗口**方法：
- 输入：过去30天的开盘价 $[x_{t-29}, x_{t-28}, ..., x_t]$
- 输出：第31天的开盘价 $x_{t+1}$

### 2. 数据预处理

#### 标准化
$$x_{normalized} = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 为均值，$\sigma$ 为标准差。

### 3. 评估指标

#### MAE（平均绝对误差）
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

#### 泛化比率
$$\text{Generalization Ratio} = \frac{\text{Test MAE}}{\text{Train MAE}}$$

## 💡 关键技术实现细节

### 1. 动态计算图管理

```python
# 每次前向传播前清理计算图
default_graph.clear_jacobi()
default_graph.reset_value()
```

### 2. 批处理实现

```python
def create_batch(X, y, batch_size):
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], y[start:end]
```

### 3. 序列处理

LSTM/RNN的序列输入处理：
```python
x_sequence = []
for t in range(seq_len):
    x_t = Variable((batch_size, input_size))
    x_t.set_value(X_batch[:, t].reshape(batch_size, input_size))
    x_sequence.append(x_t)
```

## 🎯 性能优化策略

1. **内存优化**：
   - 及时清理计算图
   - 使用生成器处理批次数据

2. **训练稳定性**：
   - 使用Adam优化器
   - 合理的学习率衰减
   - 早停机制

3. **数值稳定性**：
   - 激活函数值域限制
   - 梯度裁剪
   - 合理的参数初始化

## 📝 代码示例

### 创建和训练LSTM模型

```python
# 创建LSTM模型
lstm = LSTM(input_size=1, hidden_size=32, num_layers=1)

# 训练模型
train_losses, val_losses, output_layer = train_model(
    lstm, 'lstm', 
    (X_train, y_train), 
    (X_test, y_test),
    epochs=20, 
    batch_size=16, 
    learning_rate=0.01
)

# 预测
predictions = predict_with_model(lstm, 'lstm', X_test, output_layer)
```

## 🐛 常见问题与解决方案

1. **梯度消失/爆炸**
   - 解决：使用梯度裁剪，合理初始化参数
   - LSTM遗忘门偏置初始化为1

2. **内存不足**
   - 解决：减小batch_size，及时清理计算图

3. **训练不收敛**
   - 解决：降低学习率，使用Adam优化器

## 📊 实验结果分析
![all_models_mae_comparison](https://github.com/user-attachments/assets/34d6e706-4fe0-4073-a541-7ace5abacb55)


根据实际实验结果，各模型性能对比如下：

| 模型 | 测试MAE | 性能排名 | 说明 |
|------|---------|----------|------|
| RNN_32 | 0.7172 | 🥇 第1名 | 最佳性能，简单有效 |
| RNN_64 | 0.7207 | 🥈 第2名 | 增大容量，性能优秀 |
| LSTM_1Layer | 0.7614 | 🥉 第3名 | 单层LSTM，性能良好 |
| LSTM_Hidden32 | 0.7655 | 第4名 | 32隐藏单元基础配置 |
| LSTM_Hidden16 | 0.7773 | 第5名 | 16隐藏单元，容量较小 |
| LSTM_Hidden64 | 0.7977 | 第6名 | 64隐藏单元，容量增大 |
| RNN_Deep | 0.8079 | 第7名 | 深层RNN，复杂度提升 |
| LSTM_2Layers | 0.8548 | 第8名 | 双层LSTM，过度复杂 |
| Basic_LSTM | 0.8569 | 第9名 | 默认配置基准模型 |
| LSTM_3Layers | 0.9100 | 第10名 | 三层LSTM，明显过拟合 |
| MLP_Large | 1.1431 | 第11名 | 大型MLP，无时序建模 |
| LSTM_Attention_32 | 1.1713 | 第12名 | 注意力机制，效果不佳 |
| MLP_Small | 1.2762 | 第13名 | 小型MLP，能力有限 |
| LSTM_Attention_64 | 1.3766 | 第14名 | 64单元注意力，过度复杂 |
| MLP_Medium | 1.6401 | 第15名 | 中型MLP，性能最差 |

### 关键发现：

1. **RNN表现意外优秀**：RNN_32和RNN_64获得了最佳性能，证明了在股票预测任务中，简单的循环结构可能比复杂的LSTM更有效。

2. **LSTM层数影响**：单层LSTM (0.7614) 明显优于多层LSTM，表明过深的网络容易导致过拟合。

3. **注意力机制失效**：带注意力机制的LSTM性能较差，可能是因为股票时序数据的局部依赖性较强，全局注意力反而引入了噪声。

4. **MLP局限性明显**：所有MLP变种的性能都显著低于RNN/LSTM，证实了时序建模的重要性。

5. **隐藏单元数量**：32个隐藏单元是较优的选择，过多或过少都会影响性能。

### 训练收敛性分析：

从训练曲线可以观察到：
- RNN模型收敛快速且稳定
- 复杂模型（多层LSTM、注意力机制）容易出现训练不稳定
- MLP模型虽然收敛快，但泛化能力有限

## 👨‍💻 作者信息

- 课程：基于Python的深度学习框架实现
- 学校：深圳大学
- 时间：2025年6月

## 📄 License

本项目仅用于教学目的，请勿用于商业用途。

---

**注意**：
1. 运行前请确保 `Daily_ZX.csv` 文件在项目根目录下
2. 本项目完全基于NumPy实现，无需安装PyTorch/TensorFlow
3. 首次运行可能需要较长时间进行模型训练
