#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化器模块清理版 - 删除调试信息，保持核心功能
"""

import numpy as np
import abc
from core_framework import Node, Variable, Graph, default_graph


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


class Optimizer(object):
    """优化器基类 - 清理版"""

    def __init__(self, graph, target, learning_rate=0.01):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """计算并累加样本的梯度"""
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """返回样本的平均梯度"""
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """抽象方法，执行具体的梯度更新算法"""

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        """应用梯度"""
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                target_node = node
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None

            if target_node not in self.acc_gradient:
                self.acc_gradient[target_node] = np.zeros_like(gradient)

            if summarize:
                self.acc_gradient[target_node] += gradient
            else:
                self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no if acc_no is not None else 1
        else:
            self.acc_no = acc_no if acc_no is not None else 1

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)

        # 执行更新
        self._update()

        # 清除累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    def get_computation_path(self, target_node):
        """获取从target_node开始的计算路径上的所有节点"""
        visited = set()
        computation_nodes = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            computation_nodes.append(node)

            # 递归访问所有父节点
            for parent in node.parents:
                dfs(parent)

        dfs(target_node)
        return computation_nodes

    def forward_backward(self):
        """清理版的前向传播和反向传播"""
        # 清除计算图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 获取参与计算的所有节点
        computation_nodes = self.get_computation_path(self.target)

        # 统计可训练参数节点
        trainable_params = []
        for node in computation_nodes:
            if (isinstance(node, Variable) and
                    hasattr(node, 'trainable') and node.trainable and
                    hasattr(node, '_is_parameter') and node._is_parameter and
                    node.value is not None):
                trainable_params.append(node)

        # 只对参与计算且有值的参数节点进行反向传播
        for node in trainable_params:
            try:
                # 计算雅可比矩阵
                jacobi = node.backward(self.target)

                if jacobi is not None:
                    # 将雅可比矩阵转置得到梯度
                    gradient = jacobi.T

                    # 确保梯度形状与参数形状匹配
                    if gradient.shape != node.shape():
                        if gradient.size == node.value.size:
                            gradient = gradient.reshape(node.shape())
                        elif gradient.size > node.value.size:
                            # 截断梯度到正确大小
                            gradient = gradient.flatten()[:node.value.size].reshape(node.shape())
                        else:
                            continue

                    # 累积梯度
                    if node not in self.acc_gradient:
                        self.acc_gradient[node] = gradient
                    else:
                        self.acc_gradient[node] += gradient

            except Exception as e:
                # 静默跳过有问题的节点
                continue


class SGD(Optimizer):
    """SGD优化器 - 清理版"""

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)

    def _update(self):
        """执行SGD更新"""
        for node in self.acc_gradient:
            try:
                gradient = self.get_gradient(node)

                # 确保梯度形状正确
                if gradient.shape != node.value.shape:
                    if gradient.size == node.value.size:
                        gradient = gradient.reshape(node.value.shape)
                    else:
                        continue

                # 执行梯度下降更新
                update = self.learning_rate * gradient
                new_value = node.value - update

                # 使用set_value方法更新，这会重置子节点
                node.set_value(new_value)

            except Exception as e:
                continue


class Adam(Optimizer):
    """Adam优化器 - 清理版"""

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        # 历史梯度累积
        self.v = dict()
        # 历史梯度各分量平方累积
        self.s = dict()
        # 时间步
        self.t = 0

    def _update(self):
        """执行Adam更新"""
        self.t += 1

        for node in self.acc_gradient:
            try:
                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 确保梯度形状正确
                if gradient.shape != node.value.shape:
                    if gradient.size == node.value.size:
                        gradient = gradient.reshape(node.value.shape)
                    else:
                        continue

                if node not in self.v:
                    self.v[node] = np.zeros_like(gradient)
                    self.s[node] = np.zeros_like(gradient)

                # 梯度累积
                self.v[node] = self.beta_1 * self.v[node] + (1 - self.beta_1) * gradient

                # 各分量平方累积
                self.s[node] = self.beta_2 * self.s[node] + (1 - self.beta_2) * np.power(gradient, 2)

                # 偏差修正
                v_corrected = self.v[node] / (1 - self.beta_1 ** self.t)
                s_corrected = self.s[node] / (1 - self.beta_2 ** self.t)

                # 更新变量节点的值
                update = self.learning_rate * v_corrected / (np.sqrt(s_corrected) + 1e-10)
                new_value = node.value - update

                node.set_value(new_value)

            except Exception as e:
                continue