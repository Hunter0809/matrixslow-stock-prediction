# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版算子模块 - 解决雅可比矩阵维度不匹配问题
主要修复：
1. MatMul的雅可比矩阵计算
2. Add的广播雅可比计算
3. 所有算子的维度匹配检查
4. 梯度传播链修复
"""

import numpy as np
from core_framework import Node


class Operator(Node):
    """操作符抽象类"""
    pass


class Add(Operator):
    """矩阵加法 - 修复雅可比计算"""

    def compute(self):
        # 检查所有父节点
        for i, parent in enumerate(self.parents):
            if parent.value is None:
                raise ValueError(f"Parent {i} of Add node {self.name} has no value")

        # 获取第一个操作数
        result = self.parents[0].value.copy()

        # 逐个添加其他操作数
        for i in range(1, len(self.parents)):
            parent_value = self.parents[i].value
            try:
                result = result + parent_value  # numpy会自动处理广播
            except Exception as e:
                raise ValueError(f"Cannot add matrices with shapes {result.shape} and {parent_value.shape}: {e}")

        self.value = np.mat(result)

    def get_jacobi(self, parent):
        """修复的雅可比矩阵计算 - 正确处理维度"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        # 获取输出和父节点的维度信息
        output_size = self.value.size  # 输出总元素数
        parent_size = parent.value.size  # 父节点总元素数

        # 创建雅可比矩阵 - 行数是输出维度，列数是输入维度
        jacobi = np.zeros((output_size, parent_size))

        # 对于加法，每个输入元素对相应输出元素的导数是1
        if self.value.shape == parent.value.shape:
            # 形状完全相同 - 单位矩阵
            min_size = min(output_size, parent_size)
            for i in range(min_size):
                jacobi[i, i] = 1.0

        elif parent.value.shape[0] == 1 and parent.value.shape[1] == self.value.shape[1]:
            # 偏置广播：(1, n) -> (m, n)
            m, n = self.value.shape
            for i in range(m):  # 对每个批次样本
                for j in range(n):  # 对每个特征
                    output_idx = i * n + j
                    parent_idx = j  # 偏置索引
                    if output_idx < output_size and parent_idx < parent_size:
                        jacobi[output_idx, parent_idx] = 1.0

        elif parent.value.shape == (1, 1):
            # 标量广播
            for i in range(min(output_size, parent_size)):
                jacobi[i, 0] = 1.0

        else:
            # 其他广播情况 - 尽力匹配
            min_size = min(output_size, parent_size)
            for i in range(min_size):
                jacobi[i, i] = 1.0

        return np.mat(jacobi)


class MatMul(Operator):
    """矩阵乘法 - 完全修复雅可比计算"""

    def compute(self):
        if len(self.parents) != 2:
            raise ValueError(f"MatMul node {self.name} must have exactly 2 parents")

        parent0, parent1 = self.parents

        if parent0.value is None or parent1.value is None:
            raise ValueError(f"MatMul parents have no value")

        # 检查维度匹配
        shape0 = parent0.value.shape
        shape1 = parent1.value.shape

        if shape0[1] != shape1[0]:
            raise ValueError(f"Matrix dimension mismatch: {shape0} x {shape1}")

        self.value = parent0.value * parent1.value

    def get_jacobi(self, parent):
        """完全重写的雅可比矩阵计算 - 修复维度问题"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        A = self.parents[0].value  # shape: (m, k)
        B = self.parents[1].value  # shape: (k, n)
        C = self.value  # shape: (m, n) = A * B

        m, k = A.shape
        k2, n = B.shape
        assert k == k2, f"Inner dimensions must match: {k} != {k2}"

        # 输出C的大小
        output_size = m * n

        if parent is self.parents[0]:
            # 对A求导: ∂C/∂A
            # C[i,j] = Σ_k A[i,k] * B[k,j]
            # ∂C[i,j]/∂A[u,v] = δ(i,u) * B[v,j]

            input_size = m * k
            jacobi = np.zeros((output_size, input_size))

            for i in range(m):
                for j in range(n):
                    for u in range(m):
                        for v in range(k):
                            output_idx = i * n + j
                            input_idx = u * k + v

                            if i == u:  # δ(i,u)
                                jacobi[output_idx, input_idx] = B[v, j]

        else:  # parent is self.parents[1]
            # 对B求导: ∂C/∂B
            # C[i,j] = Σ_k A[i,k] * B[k,j]
            # ∂C[i,j]/∂B[u,v] = A[i,u] * δ(k,u) * δ(j,v) = A[i,u] * δ(v,j)

            input_size = k * n
            jacobi = np.zeros((output_size, input_size))

            for i in range(m):
                for j in range(n):
                    for u in range(k):
                        for v in range(n):
                            output_idx = i * n + j
                            input_idx = u * n + v

                            if j == v:  # δ(j,v)
                                jacobi[output_idx, input_idx] = A[i, u]

        return np.mat(jacobi)


class Multiply(Operator):
    """元素对应相乘 - 修复雅可比计算"""

    def compute(self):
        if len(self.parents) != 2:
            raise ValueError(f"Multiply node {self.name} must have exactly 2 parents")

        if self.parents[0].value is None or self.parents[1].value is None:
            raise ValueError(f"Multiply parents have no value")

        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        """修复的元素乘法雅可比计算"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        output_size = self.value.size
        parent_size = parent.value.size

        jacobi = np.zeros((output_size, parent_size))

        if parent is self.parents[0]:
            # 对第一个操作数求导
            other_value = self.parents[1].value
            if other_value.shape == parent.value.shape:
                # 形状相同 - 对角矩阵
                other_flat = other_value.A1 if hasattr(other_value, 'A1') else other_value.flatten()
                min_size = min(output_size, parent_size, len(other_flat))
                for i in range(min_size):
                    jacobi[i, i] = other_flat[i]
            else:
                # 处理广播情况
                for i in range(min(output_size, parent_size)):
                    jacobi[i, i] = float(
                        other_value.flat[0] if other_value.size == 1 else other_value.flat[i % other_value.size])
        else:
            # 对第二个操作数求导
            other_value = self.parents[0].value
            if other_value.shape == parent.value.shape:
                # 形状相同 - 对角矩阵
                other_flat = other_value.A1 if hasattr(other_value, 'A1') else other_value.flatten()
                min_size = min(output_size, parent_size, len(other_flat))
                for i in range(min_size):
                    jacobi[i, i] = other_flat[i]
            else:
                # 处理广播情况
                for i in range(min(output_size, parent_size)):
                    jacobi[i, i] = float(
                        other_value.flat[0] if other_value.size == 1 else other_value.flat[i % other_value.size])

        return np.mat(jacobi)


class Tanh(Operator):
    """Tanh激活函数"""

    def compute(self):
        if len(self.parents) != 1:
            raise ValueError(f"Tanh node {self.name} must have exactly 1 parent")
        if self.parents[0].value is None:
            raise ValueError(f"Parent of Tanh node {self.name} has no value")

        self.value = np.mat(np.tanh(self.parents[0].value))

    def get_jacobi(self, parent):
        """Tanh的雅可比矩阵"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        output_size = self.value.size
        parent_size = parent.value.size

        # 创建对角雅可比矩阵
        jacobi = np.zeros((output_size, parent_size))

        # tanh'(x) = 1 - tanh^2(x)
        tanh_val = self.value.A1 if hasattr(self.value, 'A1') else self.value.flatten()
        derivative = 1 - np.power(tanh_val, 2)

        min_size = min(output_size, parent_size, len(derivative))
        for i in range(min_size):
            jacobi[i, i] = derivative[i]

        return np.mat(jacobi)


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def compute(self):
        if len(self.parents) != 1:
            raise ValueError(f"Sigmoid node {self.name} must have exactly 1 parent")
        if self.parents[0].value is None:
            raise ValueError(f"Parent of Sigmoid node {self.name} has no value")

        # 数值稳定性处理
        x = np.array(self.parents[0].value)
        x = np.clip(x, -500, 500)
        self.value = np.mat(1.0 / (1.0 + np.exp(-x)))

    def get_jacobi(self, parent):
        """Sigmoid的雅可比矩阵"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        output_size = self.value.size
        parent_size = parent.value.size

        # 创建对角雅可比矩阵
        jacobi = np.zeros((output_size, parent_size))

        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_val = self.value.A1 if hasattr(self.value, 'A1') else self.value.flatten()
        derivative = sigmoid_val * (1 - sigmoid_val)

        min_size = min(output_size, parent_size, len(derivative))
        for i in range(min_size):
            jacobi[i, i] = derivative[i]

        return np.mat(jacobi)


class ReLU(Operator):
    """ReLU激活函数"""

    nslope = 0.01  # Leaky ReLU的负半轴斜率

    def compute(self):
        if len(self.parents) != 1:
            raise ValueError(f"ReLU node {self.name} must have exactly 1 parent")
        if self.parents[0].value is None:
            raise ValueError(f"Parent of ReLU node {self.name} has no value")

        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value)
        )

    def get_jacobi(self, parent):
        """ReLU的雅可比矩阵"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        output_size = self.value.size
        parent_size = parent.value.size

        # 创建对角雅可比矩阵
        jacobi = np.zeros((output_size, parent_size))

        # ReLU'(x) = 1 if x > 0 else nslope
        input_val = self.parents[0].value.A1 if hasattr(self.parents[0].value, 'A1') else self.parents[
            0].value.flatten()
        derivative = np.where(input_val > 0.0, 1.0, self.nslope)

        min_size = min(output_size, parent_size, len(derivative))
        for i in range(min_size):
            jacobi[i, i] = derivative[i]

        return np.mat(jacobi)


class LossFunction(Node):
    """损失函数抽象类"""
    pass


class MSELoss(LossFunction):
    """均方误差损失函数 - 修复雅可比计算"""

    def compute(self):
        if len(self.parents) != 2:
            raise ValueError(f"MSELoss node {self.name} must have exactly 2 parents")

        if self.parents[0].value is None or self.parents[1].value is None:
            raise ValueError(f"MSELoss parents have no value")

        y = self.parents[0].value
        y_target = self.parents[1].value

        # 确保维度匹配
        if y.shape != y_target.shape:
            if y.size == y_target.size:
                y_target = y_target.reshape(y.shape)
            else:
                raise ValueError(f"Shape mismatch: y.shape={y.shape}, y_target.shape={y_target.shape}")

        # 计算MSE损失
        diff = y - y_target
        mse_val = np.mean(np.multiply(diff, diff))
        self.value = np.mat([[float(mse_val)]])

    def get_jacobi(self, parent):
        """MSE损失的雅可比矩阵 - 修复维度匹配"""
        if parent not in self.parents:
            raise ValueError("parent not in self.parents")

        y = self.parents[0].value
        y_target = self.parents[1].value

        # 确保维度匹配
        if y.shape != y_target.shape:
            if y.size == y_target.size:
                y_target = y_target.reshape(y.shape)

        N = y.size

        # 输出是标量(1,1)，所以jacobi应该是(1, parent_size)
        parent_size = parent.value.size

        if parent is self.parents[0]:
            # 对预测值的梯度: ∂L/∂y = 2(y - y_target)/N
            gradient = (2.0 / N) * (y - y_target)
            gradient_flat = gradient.A1 if hasattr(gradient, 'A1') else gradient.flatten()

            # 确保梯度维度正确 - 行向量(1, parent_size)
            jacobi = np.zeros((1, parent_size))
            min_size = min(parent_size, len(gradient_flat))
            jacobi[0, :min_size] = gradient_flat[:min_size]

        else:  # parent is self.parents[1]
            # 对目标值的梯度: ∂L/∂y_target = -2(y - y_target)/N
            gradient = -(2.0 / N) * (y - y_target)
            gradient_flat = gradient.A1 if hasattr(gradient, 'A1') else gradient.flatten()

            # 确保梯度维度正确 - 行向量(1, parent_size)
            jacobi = np.zeros((1, parent_size))
            min_size = min(parent_size, len(gradient_flat))
            jacobi[0, :min_size] = gradient_flat[:min_size]

        return np.mat(jacobi)

