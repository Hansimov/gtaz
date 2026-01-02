#!/usr/bin/env python
"""GTAV 防火墙规则管理工具 - 独立入口点

此脚本作为 PyInstaller 打包的入口点，解决相对导入问题。
"""

import sys
import os

# 将 src 目录添加到路径
if getattr(sys, "frozen", False):
    # 打包后运行
    base_path = sys._MEIPASS
    sys.path.insert(0, base_path)
else:
    # 开发环境运行
    base_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(base_path, "src")
    sys.path.insert(0, src_path)

# 现在可以导入 gtaz 模块了
from gtaz.nets.blocks import main

if __name__ == "__main__":
    main()
