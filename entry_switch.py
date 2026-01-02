#!/usr/bin/env python
"""GTAV 模式切换工具 - 独立入口点

此脚本作为 PyInstaller 打包的入口点，解决相对导入问题。
在测试模式下，此入口点包含 pickup 的全部功能依赖。
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

# 预导入所有 pickup 需要的模块，确保它们被打包
# 这样即使只构建 switch.exe，也包含完整功能
import gtaz.workers.auto_pickup
import gtaz.workers.mode_switch
import gtaz.audios.cables
import gtaz.audios.detects_v3
import gtaz.audios.sounds
import gtaz.menus.commons
import gtaz.menus.interacts
import gtaz.menus.locates
import gtaz.menus.navigates
import gtaz.visions.screens
import gtaz.visions.segments
import gtaz.visions.windows
import gtaz.devices.gamepads
import gtaz.devices.keyboards
import gtaz.characters.interacts
import gtaz.nets.blocks

# 现在可以导入 gtaz 模块了
from gtaz.workers.mode_switch import main

if __name__ == "__main__":
    main()
