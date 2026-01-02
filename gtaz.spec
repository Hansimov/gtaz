# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件

用于将 GTAV 自动取货工具打包为单个 exe 文件

打包命令：
    pyinstaller gtaz.spec

或使用 build.bat 脚本
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(SPECPATH)
SRC_DIR = PROJECT_ROOT / "src"
GTAZ_DIR = SRC_DIR / "gtaz"

# vgamepad 路径（需要打包 DLL 文件）
SITE_PACKAGES = Path(sys.prefix) / "Lib" / "site-packages"
VGAMEPAD_DIR = SITE_PACKAGES / "vgamepad"

# ============= 体积优化配置 =============
# 测试模式：只构建 switch.exe（包含 pickup 全部功能）
TEST_MODE = True

# 收集数据文件
datas = [
    # 菜单模板图片
    (str(GTAZ_DIR / "menus" / "imgs"), "gtaz/menus/imgs"),
    # 音频设备切换工具
    (str(GTAZ_DIR / "audios" / "svcl.exe"), "gtaz/audios"),
    # vgamepad DLL 文件（只包含 x64）
    (str(VGAMEPAD_DIR / "win" / "vigem" / "client" / "x64" / "ViGEmClient.dll"), "vgamepad/win/vigem/client/x64"),
]

# 需要排除的二进制文件（减小体积）
# OpenCV 视频 I/O 库约 27MB，项目不需要视频功能
binaries_exclude = [
    "*opencv_videoio_ffmpeg*",     # FFmpeg 视频处理 (~27MB)
    "*opencv_videoio_msmf*",       # Windows Media Foundation
    # 注意：不能排除 libscipy_openblas，NumPy 需要它进行数学运算
    "*mklml*",                     # Intel MKL (如果有的话)
]

# 隐藏导入 - PyInstaller 可能无法自动检测的模块
hiddenimports = [
    # 第三方库
    "tclogger",
    "acto",
    "sounddevice",
    "soundfile",
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    "vgamepad",
    "psutil",
    # 项目内部模块
    "gtaz",
    "gtaz.resources",
    "gtaz.workers",
    "gtaz.workers.auto_pickup",
    "gtaz.workers.mode_switch",
    "gtaz.audios",
    "gtaz.audios.cables",
    "gtaz.audios.detects_v3",
    "gtaz.audios.sounds",
    "gtaz.nets",
    "gtaz.nets.blocks",
    "gtaz.menus",
    "gtaz.menus.commons",
    "gtaz.menus.interacts",
    "gtaz.menus.locates",
    "gtaz.menus.navigates",
    "gtaz.visions",
    "gtaz.visions.screens",
    "gtaz.visions.segments",
    "gtaz.visions.windows",
    "gtaz.devices",
    "gtaz.devices.gamepads",
    "gtaz.devices.keyboards",
    "gtaz.characters",
    "gtaz.characters.interacts",
]

# 排除不需要的模块（减小体积）
excludes = [
    # GUI 框架
    "tkinter",
    "_tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    # 科学计算（不需要）
    "matplotlib",
    "scipy",
    "pandas",
    # 深度学习（不需要）
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "timm",
    # 开发工具
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "setuptools",
    "pip",
    "wheel",
    # NumPy 不需要的子模块
    "numpy.distutils",
    "numpy.f2py",
    "numpy.testing",
    "numpy.doc",
    # PIL 不需要的插件
    "PIL.ImageTk",
    "PIL.ImageQt",
    # OpenCV 不需要的模块
    "cv2.cuda",
    "cv2.dnn",
    # 其他
    "xml.etree.ElementTree",
    "unittest",
    "doctest",
    "pydoc",
    "difflib",
    "curses",
    "multiprocessing.popen_spawn_posix",
    "multiprocessing.popen_fork",
    "multiprocessing.popen_forkserver",
]

# ============= switch.exe 配置（测试用，包含全部 pickup 功能）=============
switch_a = Analysis(
    [str(PROJECT_ROOT / "entry_switch.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# 过滤掉不需要的二进制文件以减小体积
def filter_binaries(binaries, exclude_patterns):
    """根据模式过滤二进制文件"""
    import fnmatch
    filtered = []
    excluded_size = 0
    for name, path, typecode in binaries:
        exclude = False
        # 检查完整路径名和文件名
        name_lower = name.lower().replace('\\', '/')
        basename = name_lower.split('/')[-1] if '/' in name_lower else name_lower
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            if fnmatch.fnmatch(name_lower, pattern_lower) or fnmatch.fnmatch(basename, pattern_lower):
                exclude = True
                try:
                    size = Path(path).stat().st_size
                    excluded_size += size
                    print(f"  [排除] {name} ({size / 1024 / 1024:.1f} MB)")
                except:
                    print(f"  [排除] {name}")
                break
        if not exclude:
            filtered.append((name, path, typecode))
    if excluded_size > 0:
        print(f"  [总计排除] {excluded_size / 1024 / 1024:.1f} MB")
    return filtered

print("\n=== 过滤二进制文件 ===")
switch_a.binaries = filter_binaries(switch_a.binaries, binaries_exclude)

switch_pyz = PYZ(switch_a.pure)

switch_exe = EXE(
    switch_pyz,
    switch_a.scripts,
    switch_a.binaries,
    switch_a.datas,
    [],
    name="switch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ============= 非测试模式时构建其他 exe =============
if not TEST_MODE:
    # pickup.exe 配置
    pickup_a = Analysis(
        [str(PROJECT_ROOT / "entry_pickup.py")],
        pathex=[str(SRC_DIR)],
        binaries=[],
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
    )
    pickup_a.binaries = filter_binaries(pickup_a.binaries, binaries_exclude)
    pickup_pyz = PYZ(pickup_a.pure)
    pickup_exe = EXE(
        pickup_pyz,
        pickup_a.scripts,
        pickup_a.binaries,
        pickup_a.datas,
        [],
        name="pickup",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )

    # blocks.exe 配置
    blocks_a = Analysis(
        [str(PROJECT_ROOT / "entry_blocks.py")],
        pathex=[str(SRC_DIR)],
        binaries=[],
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
    )
    blocks_a.binaries = filter_binaries(blocks_a.binaries, binaries_exclude)
    blocks_pyz = PYZ(blocks_a.pure)
    blocks_exe = EXE(
        blocks_pyz,
        blocks_a.scripts,
        blocks_a.binaries,
        blocks_a.datas,
        [],
        name="blocks",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )
