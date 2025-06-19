#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MatrixSlow核心框架 - 修复广播和梯度计算问题
"""

import numpy as np
import abc
import warnings

warnings.filterwarnings('ignore')


class Graph:
    """计算图类"""

    def __init__(self):
        self.nodes = []
        self.node_dict = {}
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)
        self.node_dict[node.name] = node

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        for node in self.nodes:
            node.reset_value(False)

    def node_count(self):
        return len(self.nodes)


# 全局默认计算图
default_graph = Graph()


class Node(object):
    """计算图节点类基类"""

    def __init__(self, *parents, **kargs):
        self.name = kargs.get('name', None)
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents)
        self.children = []
        self.value = None
        self.jacobi = None

        for parent in self.parents:
            parent.children.append(self)

        self.graph.add_node(self)

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self, **kargs):
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    def forward(self):
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def get_jacobi(self, parent):
        pass

    def backward(self, result):
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:
                        child_jacobi = child.backward(result)
                        local_jacobi = child.get_jacobi(self)

                        # 确保矩阵维度匹配
                        if child_jacobi.shape[1] == local_jacobi.shape[0]:
                            self.jacobi += child_jacobi * local_jacobi
                        else:
                            print(f"Warning: Jacobi dimension mismatch in {self.name}")
                            print(f"  child_jacobi: {child_jacobi.shape}")
                            print(f"  local_jacobi: {local_jacobi.shape}")

        return self.jacobi

    def clear_jacobi(self):
        self.jacobi = None

    def dimension(self):
        if self.value is None:
            raise ValueError(f"Node {self.name} has no value computed yet")
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        if self.value is None:
            raise ValueError(f"Node {self.name} has no value computed yet")
        return self.value.shape

    def reset_value(self, recursive=True):
        # Parameter节点不重置值
        if hasattr(self, '_is_parameter') and self._is_parameter:
            return

        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):

    def __init__(self, dim, init=False, trainable=True, **kargs):
        Node.__init__(self, **kargs)

        # 处理不同类型的dim输入
        if isinstance(dim, np.ndarray):
            self.dim = dim.shape
            self.value = np.mat(dim)
            self.trainable = trainable
            return
        elif isinstance(dim, (int, float)):
            self.dim = (int(dim), 1)
        elif isinstance(dim, (list, tuple)) and len(dim) == 1:
            self.dim = (int(dim[0]), 1)
        elif isinstance(dim, (list, tuple)) and len(dim) == 2:
            self.dim = (int(dim[0]), int(dim[1]))
        else:
            self.dim = tuple(int(x) for x in dim)

        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))
        else:
            self.value = None

        self.trainable = trainable

    def compute(self):
        if self.value is None:
            raise ValueError(f"Variable node {self.name} has no value set")

    def get_jacobi(self, parent):
        raise ValueError("Variable node has no parent!")

    def set_value(self, value):
        # 处理不同类型的输入
        if isinstance(value, np.ndarray):
            if not isinstance(value, np.matrix):
                value = np.mat(value)
        elif isinstance(value, (list, tuple)):
            value = np.mat(value)
        else:
            value = np.mat(value)

        # 检查形状匹配
        if hasattr(self, 'dim') and self.dim is not None:
            if value.shape != self.dim:
                if value.size == np.prod(self.dim):
                    value = value.reshape(self.dim)
                else:
                    raise ValueError(f"Shape mismatch: expected {self.dim}, got {value.shape}")
        else:
            self.dim = value.shape

        # 重置下游节点
        if hasattr(self, 'children'):
            for child in self.children:
                child.reset_value()

        self.value = value

    def shape(self):
        if self.value is not None:
            return self.value.shape
        elif hasattr(self, 'dim') and self.dim is not None:
            return self.dim
        else:
            raise ValueError(f"Variable node {self.name} has no shape information")

    def reset_value(self, recursive=True):
        # Parameter节点不重置值
        if hasattr(self, '_is_parameter') and self._is_parameter:
            return

        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


class Parameter(Variable):
    """可训练参数节点（修复版）"""

    def __init__(self, value, name=''):
        # 确保输入是numpy数组
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            value = np.array([[value]] if np.isscalar(value) else value)

        # 标记这是Parameter节点
        self._is_parameter = True

        # 调用父类构造函数
        super().__init__(value, init=False, trainable=True, name=name)

        # 确保值被正确设置
        if self.value is None:
            self.value = np.mat(value)

    def reset_value(self, recursive=True):
        # Parameter节点永远不重置值
        if recursive:
            for child in self.children:
                child.reset_value()