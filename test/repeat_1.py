# -*- coding: utf-8 -*-
# ModuleName: repeat_1
# Author: dongyihua543
# Time: 2024/1/3 15:24

from itertools import repeat

x = 128
# x = [5, 8]
# x = {'name': 'tim', 'age': 24}

data = tuple(repeat(x, 2))
print(data, type(data))
