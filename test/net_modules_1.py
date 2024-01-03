# -*- coding: utf-8 -*-
# ModuleName: net_modules_1
# Author: dongyihua543
# Time: 2024/1/3 14:14

from torch import nn

"""
def modules(self) -> Iterator['Module']:
    Returns an iterator over all modules in the network.
    Duplicate modules are returned only once.
"""

ln = nn.Linear(2, 2)
proj1 = nn.Conv2d(3, 768, kernel_size=[16, 16], stride=[12, 12])
proj2 = nn.Conv2d(3, 768, kernel_size=[16, 16], stride=[16, 16])
net = nn.Sequential(ln, ln, proj1, proj2)

modules = list(net.modules())
for idx, m in enumerate(net.modules()):
    print(idx, '->', m)
