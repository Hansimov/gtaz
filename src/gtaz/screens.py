"""GTAV 屏幕截取"""

import ctypes
import time
import threading

from pathlib import Path
from typing import Optional
from PIL import Image
from tclogger import TCLogger, get_now

from .windows import GTAVWindowLocator


logger = TCLogger(name="ScreenCapturer", use_prefix=True, use_prefix_ms=True)


# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR / "cache"
# 帧目录
FRAMES_DIR = CACHE_DIR / "frames"

# Windows API 常量
SRCCOPY = 0x00CC0020
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002

# 默认截图间隔（秒）
DEFAULT_INTERVAL = 0.5


class ScreenCapturer:
    """
    GTAV 游戏画面截取器。

    按照指定的时间间隔截取 GTAV 游戏窗口画面，并保存到本地文件。
    支持后台窗口截取，即使 GTAV 窗口不在前台也能正确截取。
    """

    def __init__(
        self,
        interval: Optional[float] = None,
        fps: Optional[float] = None,
        output_dir: Optional[Path] = None,
        window_locator: Optional[GTAVWindowLocator] = None,
    ):
        """
        初始化屏幕截取器。

        :param interval: 截图间隔时间（秒），优先级高于 fps
        :param fps: 每秒截图帧数，当 interval 未指定时使用
        :param output_dir: 输出目录，默认为 cache/frames
        :param window_locator: 窗口定位器，默认为 None（将自动创建）
        """
        self.interval = self._calculate_interval(interval, fps)
        self.output_dir = output_dir or FRAMES_DIR
        self.window_locator = window_locator or GTAVWindowLocator()

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 控制截图循环的标志
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None

        # 加载 Windows API
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32

    def _calculate_interval(
        self, interval: Optional[float], fps: Optional[float]
    ) -> float:
        """
        计算截图间隔时间。

        优先级: interval > fps > 默认值

        :param interval: 截图间隔时间（秒）
        :param fps: 每秒截图帧数
        :return: 计算后的间隔时间（秒）
        """
        if interval is not None:
            return interval
        if fps is not None and fps > 0:
            return 1.0 / fps
        return DEFAULT_INTERVAL

    def _generate_filename(self) -> str:
        """
        生成截图文件名，格式为 YYYY-MM-DD_HH-MM-SS-sss.png

        :return: 文件名字符串
        """
        now = get_now()
        return now.strftime("%Y-%m-%d_%H-%M-%S-") + f"{now.microsecond // 1000:03d}.png"

    def _get_window_info(self) -> Optional[tuple[int, int, int]]:
        """
        获取窗口信息（句柄和客户区尺寸）。

        :return: (hwnd, width, height) 或 None
        """
        hwnd = self.window_locator.hwnd
        if not hwnd:
            logger.warn("无法获取窗口句柄")
            return None

        client_size = self.window_locator.get_client_size()
        if not client_size:
            logger.warn("无法获取窗口客户区尺寸")
            return None

        width, height = client_size

        if width <= 0 or height <= 0:
            logger.warn(f"无效的窗口尺寸: {width}x{height}")
            return None

        return hwnd, width, height

    def _create_bitmap_info(self, width: int, height: int) -> ctypes.Array:
        """
        创建 BITMAPINFOHEADER 结构。

        :param width: 位图宽度
        :param height: 位图高度
        :return: BITMAPINFOHEADER 缓冲区
        """
        bmp_info = ctypes.create_string_buffer(40)
        # biSize
        ctypes.memmove(bmp_info, ctypes.c_int32(40).value.to_bytes(4, "little"), 4)
        # biWidth
        ctypes.memmove(
            ctypes.addressof(ctypes.c_char.from_buffer(bmp_info, 4)),
            ctypes.c_int32(width).value.to_bytes(4, "little"),
            4,
        )
        # biHeight (负值表示自上而下)
        ctypes.memmove(
            ctypes.addressof(ctypes.c_char.from_buffer(bmp_info, 8)),
            ctypes.c_int32(-height).value.to_bytes(4, "little", signed=True),
            4,
        )
        # biPlanes
        ctypes.memmove(
            ctypes.addressof(ctypes.c_char.from_buffer(bmp_info, 12)),
            ctypes.c_int16(1).value.to_bytes(2, "little"),
            2,
        )
        # biBitCount
        ctypes.memmove(
            ctypes.addressof(ctypes.c_char.from_buffer(bmp_info, 14)),
            ctypes.c_int16(32).value.to_bytes(2, "little"),
            2,
        )
        return bmp_info

    def _capture_window(self, hwnd: int, width: int, height: int) -> Optional[bytes]:
        """
        直接从窗口截取画面（支持后台窗口）。

        使用 PrintWindow API 直接从窗口获取画面，
        即使窗口被其他窗口遮挡或在后台也能正确截取。

        :param hwnd: 窗口句柄
        :param width: 窗口客户区宽度
        :param height: 窗口客户区高度
        :return: 位图原始数据，失败则返回 None
        """
        # 获取窗口 DC
        hwnd_dc = self.user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            logger.warn("无法获取窗口 DC")
            return None

        # 创建兼容 DC 和位图
        mfc_dc = self.gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = self.gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        old_bitmap = self.gdi32.SelectObject(mfc_dc, bitmap)

        # 使用 PrintWindow 截取窗口内容（支持后台窗口）
        # PW_CLIENTONLY: 只截取客户区
        # PW_RENDERFULLCONTENT: 完整渲染内容（Windows 8.1+）
        result = self.user32.PrintWindow(
            hwnd, mfc_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT
        )

        if not result:
            # 如果 PrintWindow 失败，尝试使用 BitBlt 作为后备方案
            # 注意：BitBlt 对于后台窗口可能无法正确工作
            client_dc = self.user32.GetDC(hwnd)
            self.gdi32.BitBlt(mfc_dc, 0, 0, width, height, client_dc, 0, 0, SRCCOPY)
            self.user32.ReleaseDC(hwnd, client_dc)

        # 创建位图信息头
        bmp_info = self._create_bitmap_info(width, height)

        # 创建缓冲区并获取位图数据
        buffer_size = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_size)
        self.gdi32.GetDIBits(mfc_dc, bitmap, 0, height, buffer, bmp_info, 0)

        # 清理 GDI 对象
        self.gdi32.SelectObject(mfc_dc, old_bitmap)
        self.gdi32.DeleteObject(bitmap)
        self.gdi32.DeleteDC(mfc_dc)
        self.user32.ReleaseDC(hwnd, hwnd_dc)

        return buffer.raw

    def _save_image(self, raw_data: bytes, width: int, height: int) -> Optional[Path]:
        """
        将原始位图数据保存为 PNG 文件。

        :param raw_data: BGRA 格式的原始位图数据
        :param width: 图像宽度
        :param height: 图像高度
        :return: 保存的文件路径，失败则返回 None
        """
        # 创建 PIL Image
        image = Image.frombuffer("RGBA", (width, height), raw_data, "raw", "BGRA", 0, 1)

        # 转换为 RGB（去除 Alpha 通道）
        image = image.convert("RGB")

        # 生成文件路径并保存
        filename = self._generate_filename()
        filepath = self.output_dir / filename
        image.save(filepath, "PNG")

        return filepath

    def capture_frame(self) -> Optional[Path]:
        """
        截取当前 GTAV 窗口画面。

        :return: 保存的文件路径，失败则返回 None
        """
        try:
            # 获取窗口信息
            window_info = self._get_window_info()
            if not window_info:
                return None

            hwnd, width, height = window_info

            # 截取窗口画面（支持后台窗口）
            raw_data = self._capture_window(hwnd, width, height)
            if not raw_data:
                logger.warn("截取窗口画面失败")
                return None

            # 保存图像
            filepath = self._save_image(raw_data, width, height)
            if filepath:
                logger.okay(f"截图已保存: {filepath}")

            return filepath

        except Exception as e:
            logger.err(f"截图失败: {e}")
            return None

    def _capture_loop(self):
        """截图循环（在后台线程中运行）。"""
        logger.note(f"开始截图循环，间隔: {self.interval} 秒")
        while self._running:
            self.capture_frame()
            time.sleep(self.interval)
        logger.note("截图循环已停止")

    def start(self):
        """
        开始自动截图。

        在后台线程中按照设定的间隔时间持续截图。
        """
        if self._running:
            logger.warn("截图已在运行中")
            return

        if not self.window_locator.is_window_valid():
            logger.err("GTAV 窗口未找到，无法开始截图")
            return

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.okay("自动截图已启动")

    def stop(self):
        """停止自动截图。"""
        if not self._running:
            logger.warn("截图未在运行")
            return

        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=self.interval + 1)
            self._capture_thread = None
        logger.okay("自动截图已停止")

    def is_running(self) -> bool:
        """
        检查是否正在运行自动截图。

        :return: 是否正在运行
        """
        return self._running

    def __repr__(self) -> str:
        return (
            f"ScreenCapturer("
            f"interval={self.interval}, "
            f"output_dir={self.output_dir}, "
            f"running={self._running})"
        )


def test_screen_capturer():
    """测试屏幕截取器。"""
    # 使用 fps 参数测试
    FPS = 1
    capturer = ScreenCapturer(fps=FPS)
    logger.note(f"使用 fps={FPS}，计算出的 interval={capturer.interval} 秒")

    if capturer.window_locator.is_window_valid():
        logger.note(f"截取器信息: {capturer}")

        # 单次截图测试
        logger.note("执行单次截图...")
        filepath = capturer.capture_frame()
        if filepath:
            logger.okay(f"单次截图成功: {filepath}")

        # 自动截图测试（截取 5 秒）
        logger.note("开始自动截图（5 秒）...")
        capturer.start()
        time.sleep(5)
        capturer.stop()
        logger.okay("自动截图测试完成")
    else:
        logger.err("GTAV 窗口未找到")


if __name__ == "__main__":
    test_screen_capturer()

    # python -m gtaz.screens
