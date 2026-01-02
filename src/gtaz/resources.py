"""资源路径处理模块

用于处理 PyInstaller 打包后的资源文件路径问题。
当使用 PyInstaller 的 --onefile 模式打包时，资源文件会被解压到临时目录中。
此模块提供统一的接口来获取正确的资源路径。
"""

import sys
from pathlib import Path


def is_frozen() -> bool:
    """检查当前是否在 PyInstaller 打包环境中运行"""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_base_path() -> Path:
    """获取基础路径

    - 打包后: 返回 PyInstaller 解压的临时目录
    - 开发时: 返回 src/gtaz 目录
    """
    if is_frozen():
        # PyInstaller 打包后，资源解压到 _MEIPASS 目录
        return Path(sys._MEIPASS) / "gtaz"
    else:
        # 开发环境，返回 gtaz 包目录
        return Path(__file__).parent


def get_resource_path(relative_path: str) -> Path:
    """获取资源文件的绝对路径

    :param relative_path: 相对于 gtaz 包根目录的路径
    :return: 资源文件的绝对路径

    示例:
        get_resource_path("menus/imgs/header_在线.jpg")
        get_resource_path("audios/svcl.exe")
    """
    return get_base_path() / relative_path


def get_module_dir(module_file: str) -> Path:
    """获取模块所在目录

    :param module_file: 模块的 __file__ 属性
    :return: 模块所在目录的路径

    用法:
        MODULE_DIR = get_module_dir(__file__)
    """
    if is_frozen():
        # 打包后，需要根据模块的相对路径来确定目录
        # 从 module_file 中提取相对路径
        module_path = Path(module_file)
        # 在打包环境中，__file__ 可能指向原始路径或临时目录
        # 我们需要获取相对于 gtaz 包的路径
        try:
            # 尝试找到 gtaz 在路径中的位置
            parts = module_path.parts
            if "gtaz" in parts:
                gtaz_idx = parts.index("gtaz")
                relative_parts = parts[gtaz_idx + 1 : -1]  # 不包含文件名
                return (
                    get_base_path() / Path(*relative_parts)
                    if relative_parts
                    else get_base_path()
                )
        except (ValueError, IndexError):
            pass
        return get_base_path()
    else:
        return Path(module_file).parent


def get_parent_dir(module_file: str, levels: int = 1) -> Path:
    """获取模块的上级目录

    :param module_file: 模块的 __file__ 属性
    :param levels: 向上的级数，1 表示父目录，2 表示祖父目录
    :return: 上级目录的路径

    用法:
        # 获取父目录 (相当于 Path(__file__).parent)
        MODULE_DIR = get_parent_dir(__file__, 1)

        # 获取祖父目录 (相当于 Path(__file__).parents[1])
        CACHE_DIR = get_parent_dir(__file__, 2)
    """
    if is_frozen():
        # 打包后的处理
        module_path = Path(module_file)
        try:
            parts = module_path.parts
            if "gtaz" in parts:
                gtaz_idx = parts.index("gtaz")
                relative_parts = parts[gtaz_idx + 1 : -1]  # 不包含文件名
                # 向上移动 levels 级
                if len(relative_parts) >= levels:
                    relative_parts = relative_parts[:-levels]
                    return (
                        get_base_path() / Path(*relative_parts)
                        if relative_parts
                        else get_base_path()
                    )
                else:
                    return get_base_path()
        except (ValueError, IndexError):
            pass
        return get_base_path()
    else:
        if levels == 1:
            return Path(module_file).parent
        else:
            return Path(module_file).parents[levels - 1]
