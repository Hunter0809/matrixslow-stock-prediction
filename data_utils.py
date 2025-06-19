#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具：股票数据预处理、批次生成等
"""

import numpy as np
import pandas as pd


def prepare_stock_data(file_path, seq_len=30, train_ratio=0.8):
    """准备股票数据"""

    # 首先尝试读取真实数据
    df = pd.read_csv(file_path)
    if 'open' not in df.columns:
        raise ValueError("CSV文件中没有找到'open'列")
    open_prices = df['open'].values

    # 数据清洗
    open_prices = open_prices[~np.isnan(open_prices)]

    if len(open_prices) < seq_len + 1:
        raise ValueError(f"数据长度不足，需要至少{seq_len + 1}个数据点")

    # 数据归一化
    mean = np.mean(open_prices)
    std = np.std(open_prices)
    if std == 0:
        std = 1.0
    normalized_prices = (open_prices - mean) / std

    # 创建序列数据
    X, y = [], []
    for i in range(len(normalized_prices) - seq_len):
        X.append(normalized_prices[i:i + seq_len])
        y.append(normalized_prices[i + seq_len])

    X = np.array(X)
    y = np.array(y)

    # 划分训练集和测试集
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return (X_train, y_train), (X_test, y_test), (mean, std)


def create_batch(X, y, batch_size):
    """创建批次数据"""
    n_samples = len(X)
    indices = np.arange(n_samples)  # 顺序处理，不随机打乱

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]
