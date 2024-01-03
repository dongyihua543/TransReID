# -*- coding: utf-8 -*-
# ModuleName: net_module_weight_1
# Author: dongyihua543
# Time: 2024/1/3 14:14

from torch import nn

"""
ln = nn.Linear(128, 32):
m.weight.data: 这个张量包含了实际的权重数值, 它是一个二维张量, 形状为 (output_features, input_features).
其中 output_features 是线性层的输出维度, input_features 是输入维度, 这个张量存储了线性层的权重值.
"""

ln = nn.Linear(128, 32)
proj = nn.Conv2d(3, 768, kernel_size=[16, 16], stride=[12, 12])

net = nn.Sequential(ln)
# net = nn.Sequential(proj)

modules = list(net.modules())
for idx, m in enumerate(net.modules()):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        w = m.weight.data
        print(idx, '->', m)
        print('weight:', w, w.shape)
