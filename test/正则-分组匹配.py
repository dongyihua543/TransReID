# -*- coding: utf-8 -*-
# ModuleName: 正则-分组匹配
# Author: dongyihua543
# Time: 2023/12/29 20:54

import re

"""
从字符串 img_path 中提取形如 '数字_c数字' OR '-数字_c数字' 的子串, 并将第一个数字和第二个数字分别作为两个捕获组。
"""

# img_path = "0002_c1s1_000451_03.jpg"
img_path = "-1_c5s3_069462_06.jpg"
pattern = re.compile(r'([-\d]+)_c(\d)')
tmp1 = pattern.search(img_path)
tmp2 = tmp1.groups()
pid, camid = map(int, pattern.search(img_path).groups())
print(pid)
print(camid)
