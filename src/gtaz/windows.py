"""GTAV 窗口定位"""

"""
References:
- https://github.com/shibeta/JNTMbot_python/blob/main/gta_automator/game_process.py
"""


import ctypes
from ctypes import wintypes
from typing import Optional

from tclogger import TCLogger


logger = TCLogger(name="GTAVWindowLocator", use_prefix=True)


# GTAV 增强版进程名
GTAV_PROCESS_NAME = "GTA5_Enhanced.exe"
# GTAV 增强版窗口标题
GTAV_WINDOW_TITLE = "Grand Theft Auto V"
# GTAV 增强版窗口类名
GTAV_WINDOW_CLASS_NAME = "sgaWindow"


# Windows API 常量
SW_RESTORE = 9
SW_SHOW = 5
SW_MINIMIZE = 6
SW_MAXIMIZE = 3


class GTAVWindowLocator:
    """
    用于定位 GTAV 增强版游戏窗口的类。

    提供窗口查找、激活、前置等功能。
    """

    def __init__(
        self,
        process_name: str = GTAV_PROCESS_NAME,
        window_title: str = GTAV_WINDOW_TITLE,
        window_class_name: str = GTAV_WINDOW_CLASS_NAME,
    ):
        """
        初始化窗口定位器。

        :param process_name: 进程名称
        :param window_title: 窗口标题
        :param window_class_name: 窗口类名
        """
        self.process_name = process_name
        self.window_title = window_title
        self.window_class_name = window_class_name
        self._hwnd: Optional[int] = None

        # 加载 Windows API
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

    @property
    def hwnd(self) -> Optional[int]:
        """获取缓存的窗口句柄，如果无效则重新查找。"""
        if self._hwnd and self.user32.IsWindow(self._hwnd):
            return self._hwnd
        self._hwnd = self.find_window()
        return self._hwnd

    def find_window(self) -> Optional[int]:
        """
        通过窗口类名和标题查找 GTAV 窗口。

        :return: 窗口句柄 (HWND)，未找到则返回 None
        """
        hwnd = self.user32.FindWindowW(self.window_class_name, self.window_title)
        if hwnd:
            logger.okay(f"找到 GTAV 窗口: HWND={hwnd}")
            return hwnd
        else:
            logger.warn(
                f"未找到 GTAV 窗口 (类名: {self.window_class_name}, 标题: {self.window_title})"
            )
            return None

    def is_window_valid(self) -> bool:
        """
        检查窗口是否有效（存在且可见）。

        :return: 窗口是否有效
        """
        if not self.hwnd:
            return False
        return bool(self.user32.IsWindow(self.hwnd))

    def is_window_visible(self) -> bool:
        """
        检查窗口是否可见。

        :return: 窗口是否可见
        """
        if not self.hwnd:
            return False
        return bool(self.user32.IsWindowVisible(self.hwnd))

    def is_window_foreground(self) -> bool:
        """
        检查窗口是否为前台窗口。

        :return: 窗口是否为前台窗口
        """
        if not self.hwnd:
            return False
        return self.user32.GetForegroundWindow() == self.hwnd

    def get_window_rect(self) -> Optional[tuple[int, int, int, int]]:
        """
        获取窗口的位置和大小。

        :return: (left, top, right, bottom) 或 None
        """
        if not self.hwnd:
            return None
        rect = wintypes.RECT()
        if self.user32.GetWindowRect(self.hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def get_client_rect(self) -> Optional[tuple[int, int, int, int]]:
        """
        获取窗口客户区的位置和大小。

        :return: (left, top, right, bottom) 或 None
        """
        if not self.hwnd:
            return None
        rect = wintypes.RECT()
        if self.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            # GetClientRect 返回的是相对于窗口的坐标，需要转换为屏幕坐标
            point = wintypes.POINT(0, 0)
            self.user32.ClientToScreen(self.hwnd, ctypes.byref(point))
            return (
                point.x,
                point.y,
                point.x + rect.right,
                point.y + rect.bottom,
            )
        return None

    def get_window_size(self) -> Optional[tuple[int, int]]:
        """
        获取窗口的宽度和高度。

        :return: (width, height) 或 None
        """
        rect = self.get_window_rect()
        if rect:
            return (rect[2] - rect[0], rect[3] - rect[1])
        return None

    def get_client_size(self) -> Optional[tuple[int, int]]:
        """
        获取窗口客户区的宽度和高度。

        :return: (width, height) 或 None
        """
        rect = self.get_client_rect()
        if rect:
            return (rect[2] - rect[0], rect[3] - rect[1])
        return None

    def bring_to_foreground(self) -> bool:
        """
        将窗口带到前台。

        :return: 是否成功
        """
        if not self.hwnd:
            logger.err("无法将窗口带到前台: 未找到窗口")
            return False

        try:
            # 如果窗口最小化，先恢复
            if self.user32.IsIconic(self.hwnd):
                self.user32.ShowWindow(self.hwnd, SW_RESTORE)

            # 尝试使用 SetForegroundWindow
            result = self.user32.SetForegroundWindow(self.hwnd)
            if result:
                logger.okay("成功将 GTAV 窗口带到前台")
                return True

            # 如果失败，尝试使用 AttachThreadInput 技巧
            foreground_hwnd = self.user32.GetForegroundWindow()
            if foreground_hwnd:
                foreground_thread_id = self.user32.GetWindowThreadProcessId(
                    foreground_hwnd, None
                )
                current_thread_id = self.kernel32.GetCurrentThreadId()

                if foreground_thread_id != current_thread_id:
                    self.user32.AttachThreadInput(
                        current_thread_id, foreground_thread_id, True
                    )
                    self.user32.SetForegroundWindow(self.hwnd)
                    self.user32.AttachThreadInput(
                        current_thread_id, foreground_thread_id, False
                    )

            logger.okay("成功将 GTAV 窗口带到前台")
            return True

        except Exception as e:
            logger.err(f"将窗口带到前台时出错: {e}")
            return False

    def minimize(self) -> bool:
        """
        最小化窗口。

        :return: 是否成功
        """
        if not self.hwnd:
            return False
        return bool(self.user32.ShowWindow(self.hwnd, SW_MINIMIZE))

    def maximize(self) -> bool:
        """
        最大化窗口。

        :return: 是否成功
        """
        if not self.hwnd:
            return False
        return bool(self.user32.ShowWindow(self.hwnd, SW_MAXIMIZE))

    def restore(self) -> bool:
        """
        恢复窗口（从最小化或最大化状态）。

        :return: 是否成功
        """
        if not self.hwnd:
            return False
        return bool(self.user32.ShowWindow(self.hwnd, SW_RESTORE))

    def refresh(self) -> Optional[int]:
        """
        刷新窗口句柄缓存。

        :return: 新的窗口句柄
        """
        self._hwnd = None
        return self.hwnd

    def __repr__(self) -> str:
        rect = self.get_window_rect()
        size = self.get_window_size()
        return (
            f"GTAVWindowLocator("
            f"hwnd={self._hwnd}, "
            f"valid={self.is_window_valid()}, "
            f"visible={self.is_window_visible()}, "
            f"foreground={self.is_window_foreground()}, "
            f"rect={rect}, "
            f"size={size})"
        )


def test_gtav_window_locator():
    locator = GTAVWindowLocator()

    if locator.is_window_valid():
        logger.note(f"窗口信息: {locator}")
        logger.note(f"窗口位置: {locator.get_window_rect()}")
        logger.note(f"窗口大小: {locator.get_window_size()}")
        logger.note(f"客户区位置: {locator.get_client_rect()}")
        logger.note(f"客户区大小: {locator.get_client_size()}")
        # 将窗口带到前台
        locator.bring_to_foreground()
    else:
        logger.err("GTAV 窗口未找到或无效")


if __name__ == "__main__":
    test_gtav_window_locator()

    # python -m gtaz.windows
