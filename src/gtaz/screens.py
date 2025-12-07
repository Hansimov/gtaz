"""GTAV 屏幕截取"""

import argparse
import ctypes
import json
import time
import threading

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from PIL import Image
from tclogger import TCLogger, TCLogbar, logstr, get_now

from .windows import GTAVWindowLocator
from .keyboard_actions import KeyboardActionDetector, KeyboardActionInfo
from .segments import calc_minimap_crop_region


logger = TCLogger(name="ScreenCapturer", use_prefix=True, use_prefix_ms=True)


# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR / "cache"
# 帧目录
FRAMES_DIR = CACHE_DIR / "frames"
# 动作目录（键盘触发模式）
ACTIONS_DIR = CACHE_DIR / "actions"

# Windows API 常量
SRCCOPY = 0x00CC0020
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002

# 默认截图间隔（秒）
DEFAULT_INTERVAL = 0.5

# 默认图像质量（JPEG）
DEFAULT_QUALITY = 85

# 支持的图像格式
IMAGE_FORMAT_JPEG = "JPEG"
IMAGE_FORMAT_PNG = "PNG"

# 交互式控制键
START_CAPTURE_KEY = "1"
STOP_CAPTURE_KEY = "2"
START_CAPTURE_VK = 0x31  # Virtual key code for '1'
STOP_CAPTURE_VK = 0x32  # Virtual key code for '2'


# 进度日志样式映射
PROGRESS_LOGSTR = {
    0: logstr.file,
    25: logstr.mesg,
    50: logstr.note,
    75: logstr.hint,
    100: logstr.okay,
}


def get_progress_logstr(percent: float):
    """根据百分比获取对应的日志样式函数。"""
    for threshold in sorted(PROGRESS_LOGSTR.keys(), reverse=True):
        if percent >= threshold:
            return PROGRESS_LOGSTR[threshold]
    return logstr.file


def brq(s) -> str:
    """为字符串添加单引号。"""
    return f"'{s}'"


def key_hint(s) -> str:
    """为按键添加提示样式。"""
    return logstr.hint(brq(s))


def val_mesg(s) -> str:
    """为值添加消息样式。"""
    return logstr.mesg(s)


@dataclass
class CapturedFrame:
    """
    单帧截图数据。

    用于存储截图的原始数据和元信息，以便后续批量保存。
    """

    raw_data: bytes
    """BGRA 格式的原始位图数据"""
    width: int
    """图像宽度"""
    height: int
    """图像高度"""
    timestamp: float
    """截图时间戳"""
    filename: str
    """预生成的文件名"""
    action_info: Optional[KeyboardActionInfo] = None
    """键盘动作信息（仅 KeyboardActionCapturer 使用）"""
    frame_index: int = 0
    """帧序号"""


class CapturesCache:
    """
    截图缓存管理器。

    将截图数据缓存在内存中，在时间窗口结束时批量保存到文件。
    支持普通截图和带键盘动作信息的截图。
    """

    def __init__(
        self,
        output_dir: Path,
        image_format: str = IMAGE_FORMAT_JPEG,
        quality: int = DEFAULT_QUALITY,
    ):
        """
        初始化缓存管理器。

        :param output_dir: 输出目录
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"
        :param quality: JPEG 质量（1-100），默认 85
        """
        self.output_dir = output_dir
        self.image_format = image_format.upper()
        self.quality = max(1, min(100, quality))

        # 缓存的帧列表
        self._frames: list[CapturedFrame] = []
        # 帧计数器（避免每次计算 len）
        self._frame_count: int = 0
        # 线程锁，确保线程安全
        self._lock = threading.Lock()

    def add_frame(self, frame: CapturedFrame):
        """
        添加一帧到缓存。

        :param frame: 截图帧数据
        """
        with self._lock:
            self._frames.append(frame)
            self._frame_count += 1

    def get_frame_count(self) -> int:
        """
        获取缓存中的帧数。

        :return: 帧数
        """
        with self._lock:
            return self._frame_count

    def clear(self):
        """
        清空缓存。
        """
        with self._lock:
            self._frames.clear()
            self._frame_count = 0

    def _save_single_image(self, frame: CapturedFrame) -> Path:
        """
        保存单帧图像到文件。

        :param frame: 截图帧数据
        :return: 保存的文件路径
        """
        # 创建 PIL Image
        image = Image.frombuffer(
            "RGBA", (frame.width, frame.height), frame.raw_data, "raw", "BGRA", 0, 1
        )

        # 生成文件路径
        filepath = self.output_dir / frame.filename

        # 根据格式保存
        if self.image_format == IMAGE_FORMAT_JPEG:
            image = image.convert("RGB")
            image.save(filepath, "JPEG", quality=self.quality, optimize=False)
        else:
            image = image.convert("RGB")
            image.save(filepath, "PNG", compress_level=1)

        return filepath

    def _save_action_info(
        self,
        json_filepath: Path,
        frame: CapturedFrame,
        image_filepath: Path,
    ) -> bool:
        """
        保存键盘动作信息到 JSON 文件。

        :param json_filepath: JSON 文件路径
        :param frame: 截图帧数据
        :param image_filepath: 对应的图像文件路径
        :return: 是否保存成功
        """
        if not frame.action_info:
            return False

        # 构建按键信息列表
        keys_list = []
        for key_state in frame.action_info.key_states.values():
            # 计算按键持续时间
            press_duration = None
            if key_state.press_time is not None:
                if key_state.release_time is not None:
                    press_duration = key_state.release_time - key_state.press_time
                else:
                    # 如果还未释放，使用 action_info.timestamp 计算（与 key_state 同时记录）
                    press_duration = frame.action_info.timestamp - key_state.press_time
                # 确保 press_duration 不为负数（可能由于对象引用被后续修改导致）
                if press_duration < 0:
                    press_duration = 0.0

            keys_list.append(
                {
                    "key_name": key_state.key_name,
                    "is_pressed": key_state.is_pressed,
                    "press_at": key_state.press_time,
                    "press_duration": press_duration,
                    "release_at": key_state.release_time,
                }
            )

        data = {
            "time": {
                "timestamp": frame.action_info.timestamp,
                "datetime": frame.action_info.datetime_str,
            },
            "has_action": frame.action_info.has_action,
            "keys": keys_list,
            "frame": {
                "index": frame.frame_index,
                "width": frame.width,
                "height": frame.height,
            },
            "file": {
                "image": image_filepath.name,
                "json": json_filepath.name,
                "directory": str(self.output_dir),
            },
        }

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    def flush(self, verbose: bool = True) -> int:
        """
        将缓存中的所有帧保存到文件。

        :param verbose: 是否打印保存日志
        :return: 成功保存的帧数
        """
        with self._lock:
            frames_to_save = self._frames.copy()
            self._frames.clear()
            self._frame_count = 0

        if not frames_to_save:
            return 0

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.note(f"开始保存 {len(frames_to_save)} 帧到文件...")

        bar = TCLogbar(total=len(frames_to_save), desc="* 保存截图")
        saved_count = 0
        for i, frame in enumerate(frames_to_save):
            filepath = self._save_single_image(frame)
            if filepath:
                saved_count += 1

                # 如果有键盘动作信息，保存 JSON 文件
                if frame.action_info:
                    json_filepath = filepath.with_suffix(".json")
                    self._save_action_info(json_filepath, frame, filepath)

                if verbose:
                    bar.update(1)
        bar.update(flush=True)
        print()

        if verbose:
            logger.okay(f"保存完成，共 {saved_count}/{len(frames_to_save)} 帧")

        return saved_count

    def __len__(self) -> int:
        return self.get_frame_count()

    def __repr__(self) -> str:
        return (
            f"CapturesCache("
            f"frames={len(self)}, "
            f"format={self.image_format}, "
            f"output_dir={self.output_dir})"
        )


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
        image_format: str = IMAGE_FORMAT_JPEG,
        quality: int = DEFAULT_QUALITY,
        use_cache: bool = True,
        minimap_only: bool = False,
    ):
        """
        初始化屏幕截取器。

        :param interval: 截图间隔时间（秒），优先级高于 fps
        :param fps: 每秒截图帧数，当 interval 未指定时使用
        :param output_dir: 输出目录，默认为 cache/frames
        :param window_locator: 窗口定位器，默认为 None（将自动创建）
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"，默认 JPEG（更快更小）
        :param quality: JPEG 质量（1-100），默认 85
        :param use_cache: 是否使用缓存模式（先缓存后批量保存），默认 True
        :param minimap_only: 是否仅截取小地图区域，默认 False
        """
        self.interval = self._calculate_interval(interval, fps)
        self.window_locator = window_locator or GTAVWindowLocator()
        self.image_format = image_format.upper()
        self.quality = max(1, min(100, quality))
        self.use_cache = use_cache
        self.minimap_only = minimap_only

        # 小地图裁剪区域（首次截图时计算）
        self._minimap_crop_region: Optional[tuple[int, int, int, int]] = None

        # 生成基于启动时间的会话目录
        if output_dir is None:
            session_name = get_now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = FRAMES_DIR / session_name
        else:
            self.output_dir = output_dir

        # 初始化缓存管理器（minimap_crop_region 在首次截图时设置）
        self._cache = CapturesCache(
            output_dir=self.output_dir,
            image_format=self.image_format,
            quality=self.quality,
        )

        # 帧计数器
        self._frame_count = 0

        # 定时器（用于动态时间补偿）
        self._next_tick_time: float = 0.0

        # 控制截图循环的标志
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None

        # 加载 Windows API
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32

        # 缓存的 GDI 资源（用于复用）
        self._cached_width: int = 0
        self._cached_height: int = 0
        self._cached_dc = None
        self._cached_bitmap = None
        self._cached_bmp_info = None

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

    def _generate_filename(self, frame_index: int) -> str:
        """
        生成截图文件名，格式为 YYYY-MM-DD_HH-MM-SS-sss_<frame_idx>.ext

        :param frame_index: 帧索引（0000-9999）
        :return: 文件名字符串
        """
        now = get_now()
        ext = "jpg" if self.image_format == IMAGE_FORMAT_JPEG else "png"
        return (
            now.strftime("%Y-%m-%d_%H-%M-%S-")
            + f"{now.microsecond // 1000:03d}_{frame_index:04d}.{ext}"
        )

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

    def _ensure_gdi_resources(self, hwnd_dc: int, width: int, height: int):
        """
        确保 GDI 资源已创建并与当前尺寸匹配。

        :param hwnd_dc: 窗口 DC
        :param width: 宽度
        :param height: 高度
        """
        # 如果尺寸变化，需要重新创建资源
        if (
            self._cached_width != width
            or self._cached_height != height
            or self._cached_dc is None
        ):
            self._release_gdi_resources()

            self._cached_width = width
            self._cached_height = height
            self._cached_dc = self.gdi32.CreateCompatibleDC(hwnd_dc)
            self._cached_bitmap = self.gdi32.CreateCompatibleBitmap(
                hwnd_dc, width, height
            )
            self._cached_bmp_info = self._create_bitmap_info(width, height)

    def _release_gdi_resources(self):
        """释放缓存的 GDI 资源。"""
        if self._cached_bitmap:
            self.gdi32.DeleteObject(self._cached_bitmap)
            self._cached_bitmap = None
        if self._cached_dc:
            self.gdi32.DeleteDC(self._cached_dc)
            self._cached_dc = None
        self._cached_bmp_info = None
        self._cached_width = 0
        self._cached_height = 0

    def _crop_raw_data(
        self,
        raw_data: bytes,
        src_width: int,
        src_height: int,
        crop_region: tuple[int, int, int, int],
    ) -> tuple[bytes, int, int]:
        """
        在字节级别裁剪 BGRA 原始数据。

        :param raw_data: BGRA 格式的原始位图数据
        :param src_width: 源图像宽度
        :param src_height: 源图像高度
        :param crop_region: 裁剪区域 (left, top, right, bottom)
        :return: (裁剪后的数据, 新宽度, 新高度)
        """
        left, top, right, bottom = crop_region
        crop_width = right - left
        crop_height = bottom - top
        bytes_per_pixel = 4  # BGRA
        src_stride = src_width * bytes_per_pixel
        crop_stride = crop_width * bytes_per_pixel

        # 逐行提取裁剪区域的数据
        cropped_rows = []
        for y in range(top, bottom):
            row_start = y * src_stride + left * bytes_per_pixel
            row_end = row_start + crop_stride
            cropped_rows.append(raw_data[row_start:row_end])

        return b"".join(cropped_rows), crop_width, crop_height

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

        # 确保 GDI 资源已准备好
        self._ensure_gdi_resources(hwnd_dc, width, height)

        # 选择位图到 DC
        old_bitmap = self.gdi32.SelectObject(self._cached_dc, self._cached_bitmap)

        # 使用 PrintWindow 截取窗口内容（支持后台窗口）
        result = self.user32.PrintWindow(
            hwnd, self._cached_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT
        )

        if not result:
            # 如果 PrintWindow 失败，尝试使用 BitBlt 作为后备方案
            client_dc = self.user32.GetDC(hwnd)
            self.gdi32.BitBlt(
                self._cached_dc, 0, 0, width, height, client_dc, 0, 0, SRCCOPY
            )
            self.user32.ReleaseDC(hwnd, client_dc)

        # 创建缓冲区并获取位图数据
        buffer_size = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_size)
        self.gdi32.GetDIBits(
            self._cached_dc,
            self._cached_bitmap,
            0,
            height,
            buffer,
            self._cached_bmp_info,
            0,
        )

        # 恢复旧位图
        self.gdi32.SelectObject(self._cached_dc, old_bitmap)

        # 释放窗口 DC
        self.user32.ReleaseDC(hwnd, hwnd_dc)

        return buffer.raw

    def _save_image(self, raw_data: bytes, width: int, height: int) -> Optional[Path]:
        """
        将原始位图数据保存为图像文件（直接保存，不使用缓存）。

        :param raw_data: BGRA 格式的原始位图数据
        :param width: 图像宽度
        :param height: 图像高度
        :return: 保存的文件路径，失败则返回 None
        """
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 更新帧计数（直接保存模式需要手动计数）
        self._frame_count += 1

        # 创建 PIL Image
        image = Image.frombuffer("RGBA", (width, height), raw_data, "raw", "BGRA", 0, 1)

        # 生成文件路径
        filename = self._generate_filename(self._frame_count)
        filepath = self.output_dir / filename

        # 根据格式保存
        if self.image_format == IMAGE_FORMAT_JPEG:
            # JPEG 不支持 Alpha，转换为 RGB
            image = image.convert("RGB")
            image.save(filepath, "JPEG", quality=self.quality, optimize=False)
        else:
            # PNG 格式，使用快速压缩
            image = image.convert("RGB")
            image.save(filepath, "PNG", compress_level=1)

        return filepath

    def _cache_frame(
        self,
        raw_data: bytes,
        width: int,
        height: int,
        action_info: Optional[KeyboardActionInfo] = None,
    ) -> str:
        """
        将帧数据添加到缓存。

        :param raw_data: BGRA 格式的原始位图数据
        :param width: 图像宽度
        :param height: 图像高度
        :param action_info: 键盘动作信息（可选）
        :return: 预生成的文件名
        """
        self._frame_count += 1
        filename = self._generate_filename(self._frame_count)

        frame = CapturedFrame(
            raw_data=raw_data,
            width=width,
            height=height,
            timestamp=time.time(),
            filename=filename,
            action_info=action_info,
            frame_index=self._frame_count,
        )

        self._cache.add_frame(frame)
        return filename

    def _ensure_minimap_crop_region(self, width: int, height: int):
        """
        确保小地图裁剪区域已计算。

        仅在首次截图时根据窗口分辨率计算，后续复用。

        :param width: 窗口宽度
        :param height: 窗口高度
        """
        if self.minimap_only and self._minimap_crop_region is None:
            self._minimap_crop_region = calc_minimap_crop_region(width, height)
            logger.note(
                f"小地图裁剪区域已计算: {self._minimap_crop_region} "
                f"(窗口: {width}x{height})"
            )

    def capture_frame(self, verbose: bool = True) -> Optional[str]:
        """
        截取当前 GTAV 窗口画面。

        根据 use_cache 参数决定是直接保存还是缓存到内存。

        :param verbose: 是否打印保存日志
        :return: 文件名（缓存模式）或保存的文件路径（直接保存模式），失败则返回 None
        """
        # 获取窗口信息
        window_info = self._get_window_info()
        if not window_info:
            return None

        hwnd, width, height = window_info

        # 确保小地图裁剪区域已计算（仅首次）
        self._ensure_minimap_crop_region(width, height)

        # 截取窗口画面（支持后台窗口）
        raw_data = self._capture_window(hwnd, width, height)
        if not raw_data:
            logger.warn("截取窗口画面失败")
            return None

        # 如果仅截取小地图，在字节级别裁剪原始数据
        frame_width, frame_height = width, height
        if self._minimap_crop_region:
            raw_data, frame_width, frame_height = self._crop_raw_data(
                raw_data, width, height, self._minimap_crop_region
            )

        if self.use_cache:
            # 缓存模式：添加到缓存
            filename = self._cache_frame(raw_data, frame_width, frame_height)
            if verbose:
                cached_count = self.get_cached_frame_count()
                logger.okay(f"已截取并缓存 {cached_count} 帧")
            return filename
        else:
            # 直接保存模式
            filepath = self._save_image(raw_data, frame_width, frame_height)
            if filepath and verbose:
                logger.okay(f"截图已保存: {filepath}")
            return str(filepath) if filepath else None

    def flush_cache(self, verbose: bool = True) -> int:
        """
        将缓存中的所有帧保存到文件。

        :param verbose: 是否打印保存日志
        :return: 成功保存的帧数
        """
        saved_count = self._cache.flush(verbose=verbose)
        # 保存完成后重置帧计数
        self._frame_count = 0
        return saved_count

    def get_cached_frame_count(self) -> int:
        """
        获取缓存中的帧数。

        :return: 帧数
        """
        return self._cache.get_frame_count()

    def reset_tick(self):
        """重置定时器，将下一次 tick 时间设为当前时间。"""
        self._next_tick_time = time.time()

    def wait_next_tick(self):
        """
        等待到下一个 tick 时间点。

        使用动态时间补偿，确保帧率稳定。
        """
        self._next_tick_time += self.interval
        sleep_time = self._next_tick_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _capture_loop(self):
        """截图循环（在后台线程中运行）。"""
        logger.note(f"开始截图循环，间隔: {self.interval} 秒")
        self.reset_tick()
        while self._running:
            self.capture_frame()
            self.wait_next_tick()
        logger.note("截图循环已停止")

    def start(self):
        """
        开始连续截图。

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
        logger.okay("连续截图已启动")

    def stop(self, flush: bool = True):
        """
        停止连续截图。

        :param flush: 是否在停止时保存缓存中的帧，默认 True
        """
        if not self._running:
            logger.warn("截图未在运行")
            return

        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=self.interval + 1)
            self._capture_thread = None

        # 保存缓存中的帧
        if flush and self.use_cache:
            self.flush_cache()

        # 释放 GDI 资源
        self._release_gdi_resources()
        logger.okay("连续截图已停止")

    def __del__(self):
        """析构函数，确保释放 GDI 资源。"""
        self._release_gdi_resources()

    def is_running(self) -> bool:
        """
        检查是否正在运行连续截图。

        :return: 是否正在运行
        """
        return self._running

    def __repr__(self) -> str:
        return (
            f"ScreenCapturer("
            f"interval={self.interval}, "
            f"format={self.image_format}, "
            f"quality={self.quality}, "
            f"output_dir={self.output_dir}, "
            f"use_cache={self.use_cache}, "
            f"minimap_only={self.minimap_only}, "
            f"cached_frames={len(self._cache)}, "
            f"running={self._running})"
        )


class KeyboardActionCapturer(ScreenCapturer):
    """
    基于键盘动作触发的截图器。

    仅在检测到键盘输入时才进行截图，并保存键盘动作信息到 JSON 文件。
    """

    def __init__(
        self,
        interval: Optional[float] = None,
        fps: Optional[float] = None,
        output_dir: Optional[Path] = None,
        window_locator: Optional[GTAVWindowLocator] = None,
        image_format: str = IMAGE_FORMAT_JPEG,
        quality: int = DEFAULT_QUALITY,
        game_keys_only: bool = True,
        use_cache: bool = True,
        minimap_only: bool = False,
    ):
        """
        初始化键盘动作截图器。

        :param interval: 截图间隔时间（秒），优先级高于 fps
        :param fps: 每秒截图帧数，当 interval 未指定时使用
        :param output_dir: 输出目录，默认为 cache/actions
        :param window_locator: 窗口定位器，默认为 None（将自动创建）
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"，默认 JPEG（更快更小）
        :param quality: JPEG 质量（1-100），默认 85
        :param game_keys_only: 是否只监控 GTAV 游戏常用按键，默认 True
        :param use_cache: 是否使用缓存模式（先缓存后批量保存），默认 True
        :param minimap_only: 是否仅截取小地图区域，默认 False
        """
        # 生成基于启动时间的会话目录（使用 actions 目录）
        if output_dir is None:
            session_name = get_now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = ACTIONS_DIR / session_name

        super().__init__(
            interval=interval,
            fps=fps,
            output_dir=output_dir,
            window_locator=window_locator,
            image_format=image_format,
            quality=quality,
            use_cache=use_cache,
            minimap_only=minimap_only,
        )

        # 初始化键盘动作检测器
        self.keyboard_detector = KeyboardActionDetector(game_keys_only=game_keys_only)

    def capture_frame_with_action(
        self, action_info: KeyboardActionInfo, verbose: bool = True
    ) -> Optional[str]:
        """
        截取当前 GTAV 窗口画面，并记录键盘动作信息。

        根据 use_cache 参数决定是直接保存还是缓存到内存。

        :param action_info: 键盘动作信息
        :param verbose: 是否打印保存日志
        :return: 文件名（缓存模式）或保存的文件路径（直接保存模式），失败则返回 None
        """
        # 获取窗口信息
        window_info = self._get_window_info()
        if not window_info:
            return None

        hwnd, width, height = window_info

        # 确保小地图裁剪区域已计算（仅首次）
        self._ensure_minimap_crop_region(width, height)

        # 截取窗口画面（支持后台窗口）
        raw_data = self._capture_window(hwnd, width, height)
        if not raw_data:
            logger.warn("截取窗口画面失败")
            return None

        # 如果仅截取小地图，在字节级别裁剪原始数据
        frame_width, frame_height = width, height
        if self._minimap_crop_region:
            raw_data, frame_width, frame_height = self._crop_raw_data(
                raw_data, width, height, self._minimap_crop_region
            )

        keys_str = ", ".join(action_info.pressed_keys)

        if self.use_cache:
            # 缓存模式：添加到缓存（包含动作信息）
            filename = self._cache_frame(
                raw_data, frame_width, frame_height, action_info
            )
            if verbose:
                cached_count = self.get_cached_frame_count()
                logger.okay(f"已截取并缓存 {cached_count} 帧 (按键: {keys_str})")
            return filename
        else:
            # 直接保存模式
            filepath = self._save_image(raw_data, frame_width, frame_height)
            if not filepath:
                return None

            # 更新帧计数
            self._frame_count += 1

            # 保存 JSON 文件
            json_filepath = filepath.with_suffix(".json")
            self._cache._save_action_info(
                json_filepath,
                CapturedFrame(
                    raw_data=raw_data,
                    width=frame_width,
                    height=frame_height,
                    timestamp=time.time(),
                    filename=filepath.name,
                    action_info=action_info,
                    frame_index=self._frame_count,
                ),
                filepath,
            )

            if verbose:
                logger.okay(f"截图已保存: {filepath.name} (按键: {keys_str})")

            return str(filepath)

    def _capture_loop(self):
        """截图循环（在后台线程中运行）。"""
        logger.note(f"开始键盘触发截图循环，检测间隔: {self.interval} 秒")
        self.reset_tick()
        while self._running:
            action_info = self.keyboard_detector.detect()
            if action_info.has_action:
                self.capture_frame_with_action(action_info)
            self.wait_next_tick()
        logger.note("键盘触发截图循环已停止")

    def __repr__(self) -> str:
        return (
            f"KeyboardActionCapturer("
            f"interval={self.interval}, "
            f"format={self.image_format}, "
            f"quality={self.quality}, "
            f"output_dir={self.output_dir}, "
            f"use_cache={self.use_cache}, "
            f"cached_frames={len(self._cache)}, "
            f"running={self._running}, "
            f"frame_count={self._frame_count})"
        )


class InteractiveController:
    """
    交互式控制器。

    使用 START_CAPTURE_KEY 键启动截图，STOP_CAPTURE_KEY 键停止截图。
    通过边沿检测避免重复触发。
    """

    def __init__(self, capturer: Union[ScreenCapturer, KeyboardActionCapturer]):
        """
        初始化交互式控制器。

        :param capturer: 截图器实例
        """
        self.capturer = capturer
        # 初始化键盘检测器（只监控启动和停止键）
        self.control_detector = KeyboardActionDetector(
            monitored_keys=[START_CAPTURE_VK, STOP_CAPTURE_VK]
        )
        self.capture_active = False
        # 记录上一次按键状态，用于边沿检测
        self._last_start_key_pressed = False
        self._last_stop_key_pressed = False

    def is_active(self) -> bool:
        """
        检查截图是否正在运行。

        :return: 是否正在运行
        """
        return self.capture_active

    def start_capture(self, mode_desc: str = "截图"):
        """
        启动截图。

        :param mode_desc: 模式描述（用于日志显示）
        """
        if not self.capture_active:
            self.capture_active = True
            self.capturer.start()
            logger.okay(
                f"{mode_desc}已开始（按 {key_hint(STOP_CAPTURE_KEY)} {val_mesg('停止截图')}）"
            )

    def stop_capture(self):
        """停止截图并保存。"""
        if self.capture_active:
            self.capture_active = False
            self.capturer.stop(flush=True)
            logger.okay("截图已停止")

    def process_control_keys(self, mode_desc: str = "截图") -> bool:
        """
        处理控制键（START_CAPTURE_KEY 启动，STOP_CAPTURE_KEY 停止）。

        使用边沿检测避免重复触发。

        :param mode_desc: 模式描述（用于日志显示）
        :return: 是否应该继续循环
        """
        action_info = self.control_detector.detect()

        # 检查启动键
        start_key_pressed = START_CAPTURE_KEY in action_info.pressed_keys
        if start_key_pressed and not self._last_start_key_pressed:
            # 按键从未按下变为按下（上升沿）
            self.start_capture(mode_desc)
        self._last_start_key_pressed = start_key_pressed

        # 检查停止键
        stop_key_pressed = STOP_CAPTURE_KEY in action_info.pressed_keys
        if stop_key_pressed and not self._last_stop_key_pressed:
            # 按键从未按下变为按下（上升沿）
            self.stop_capture()
        self._last_stop_key_pressed = stop_key_pressed

        return True

    def run(self, mode_desc: str = "截图"):
        """
        运行交互式控制循环。

        :param mode_desc: 模式描述（用于日志显示）
        """
        logger.note("交互式截图模式已启动")
        logger.note(
            f"按 {key_hint(START_CAPTURE_KEY)} {val_mesg('开始截图')}，"
            f"按 {key_hint(STOP_CAPTURE_KEY)} {val_mesg('停止截图')}，"
            f"按 {key_hint('Ctrl+C')} {val_mesg('退出')}"
        )

        try:
            while True:
                self.process_control_keys(mode_desc)
                time.sleep(0.05)  # 50ms 检测间隔

        except KeyboardInterrupt:
            logger.note("\n检测到 Ctrl+C，正在退出...")
            if self.capture_active:
                self.stop_capture()

        logger.note("交互式截图模式已退出")


class CapturerRunner:
    """
    截图器运行器。

    封装各种运行模式的逻辑，包括单帧、连续截图、交互式控制等。
    """

    def __init__(
        self,
        capturer: Union[ScreenCapturer, KeyboardActionCapturer],
        keyboard_trigger: bool = False,
    ):
        """
        初始化截图器运行器。

        :param capturer: 截图器实例
        :param keyboard_trigger: 是否为键盘触发模式
        """
        self.capturer = capturer
        self.keyboard_trigger = keyboard_trigger
        self.mode_desc = "键盘触发截图" if keyboard_trigger else "连续截图"

    def _print_config(self, interactive: bool = False):
        """
        打印配置信息。

        :param interactive: 是否为交互式模式
        """
        minimap_str = "（仅小地图）" if self.capturer.minimap_only else ""
        mode_str = "键盘触发模式，" if self.keyboard_trigger else ""
        interactive_str = "交互式" if interactive else ""
        logger.note(
            f"{interactive_str}{mode_str}fps={1/self.capturer.interval:.1f}，"
            f"interval={round(self.capturer.interval, 2)}s，"
            f"use_cache={self.capturer.use_cache}{minimap_str}"
        )

    def _validate_window(self) -> bool:
        """
        验证窗口是否有效。

        :return: 窗口是否有效
        """
        if not self.capturer.window_locator.is_window_valid():
            logger.err("GTAV 窗口未找到")
            return False
        return True

    def run_single_frame(self):
        """运行单帧截图模式。"""
        if self.keyboard_trigger:
            logger.note("执行单次截图，等待键盘输入...")
            while True:
                action_info = self.capturer.keyboard_detector.detect()
                if action_info.has_action:
                    filepath = self.capturer.capture_frame_with_action(
                        action_info, verbose=True
                    )
                    if filepath:
                        logger.okay(f"单次截图成功: {filepath}")
                    break
                time.sleep(self.capturer.interval)
        else:
            logger.note("执行单次截图...")
            filepath = self.capturer.capture_frame(verbose=True)
            if filepath:
                logger.okay(f"单次截图成功: {filepath}")

    def run_interactive(self):
        """运行交互式控制模式。"""
        controller = InteractiveController(self.capturer)
        controller.run(self.mode_desc)

    def _setup_continuous_mode(
        self, duration: float
    ) -> tuple[float, bool, Optional[KeyboardActionDetector], bool]:
        """
        设置连续截图模式的参数。

        :param duration: 持续时间（秒）
        :return: (实际持续时间, 是否使用停止键, 停止键检测器, 上次停止键状态)
        """
        max_duration = 600  # 10分钟

        if duration == 0:
            # 持续模式
            duration = max_duration
            logger.note(
                f"{self.mode_desc}（持续模式，最大 {max_duration} 秒，"
                f"按 {key_hint(STOP_CAPTURE_KEY)} {val_mesg('停止截图')}），缓存模式..."
            )
            stop_detector = KeyboardActionDetector(monitored_keys=[STOP_CAPTURE_VK])
            return duration, True, stop_detector, False
        else:
            # 定时模式
            logger.note(f"{self.mode_desc}（{duration} 秒），缓存模式...")
            return duration, False, None, False

    def _check_stop_key(
        self,
        use_stop_key: bool,
        stop_detector: Optional[KeyboardActionDetector],
        last_stop_key_pressed: bool,
    ) -> tuple[bool, bool]:
        """
        检查是否按下停止键。

        :param use_stop_key: 是否启用停止键检测
        :param stop_detector: 停止键检测器
        :param last_stop_key_pressed: 上次停止键状态
        :return: (是否应该停止, 新的停止键状态)
        """
        if not use_stop_key or stop_detector is None:
            return False, last_stop_key_pressed

        action_info = stop_detector.detect()
        stop_key_pressed = STOP_CAPTURE_KEY in action_info.pressed_keys

        if stop_key_pressed and not last_stop_key_pressed:
            logger.note(f"检测到 '{STOP_CAPTURE_KEY}' 键，停止截图...")
            return True, stop_key_pressed

        return False, stop_key_pressed

    def _capture_frame(self) -> tuple[Optional[str], str]:
        """
        根据模式执行一次截图。

        :return: (文件名, 额外信息)
        """
        if self.keyboard_trigger:
            action_info = self.capturer.keyboard_detector.detect()
            if action_info.has_action:
                filename = self.capturer.capture_frame_with_action(
                    action_info, verbose=False
                )
                extra_info = f" (按键: {', '.join(action_info.pressed_keys)})"
                return filename, extra_info
            return None, ""
        else:
            filename = self.capturer.capture_frame(verbose=False)
            return filename, ""

    def _log_progress(
        self, elapsed: float, duration: float, cached_count: int, extra_info: str
    ):
        """
        记录进度日志。

        :param elapsed: 已用时间
        :param duration: 总持续时间
        :param cached_count: 已缓存帧数
        :param extra_info: 额外信息
        """
        percent = (elapsed / duration) * 100
        progress_logstr = get_progress_logstr(percent)
        progress_str = progress_logstr(
            f"({percent:5.1f}%) [{elapsed:.1f}/{duration:.1f}]"
        )
        logger.okay(f"{progress_str} 已缓存 {cached_count} 帧{extra_info}")

    def _finalize_capture(self):
        """完成截图并保存。"""
        frame_count = self.capturer.get_cached_frame_count()
        logger.note(f"截图完成，共截取 {frame_count} 帧，开始保存...")
        saved_count = self.capturer.flush_cache(verbose=True)
        logger.okay(f"连续截图完成，共保存 {saved_count} 帧")

    def run_continuous(self, duration: float):
        """
        运行连续截图模式。

        :param duration: 持续时间（秒），0表示持续模式（最大10分钟）
        """
        # 设置参数
        (
            duration,
            use_stop_key,
            stop_detector,
            last_stop_key_pressed,
        ) = self._setup_continuous_mode(duration)

        # 开始截图循环
        start_time = time.time()
        self.capturer.reset_tick()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # 检查停止键
            should_stop, last_stop_key_pressed = self._check_stop_key(
                use_stop_key, stop_detector, last_stop_key_pressed
            )
            if should_stop:
                break

            # 执行截图
            filename, extra_info = self._capture_frame()

            # 记录进度
            if filename:
                cached_count = self.capturer.get_cached_frame_count()
                self._log_progress(elapsed, duration, cached_count, extra_info)

            self.capturer.wait_next_tick()

        # 完成并保存
        self._finalize_capture()

    def run(
        self,
        single: bool = False,
        duration: float = 60,
        interactive: bool = False,
    ):
        """
        运行截图器。

        :param single: 是否只截取单帧
        :param duration: 连续截图持续时间（秒）
        :param interactive: 是否为交互式控制模式
        """
        # 打印配置信息
        self._print_config(interactive=interactive)

        # 验证窗口
        if not self._validate_window():
            return

        logger.note(f"截取器信息: {self.capturer}")

        # 根据模式运行
        if interactive:
            self.run_interactive()
        elif single:
            self.run_single_frame()
        else:
            self.run_continuous(duration)


class ScreenCapturerArgParser:
    """屏幕截取器命令行参数解析器。"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GTAV 屏幕截取器")
        self._add_arguments()

    def _add_arguments(self):
        """添加命令行参数。"""
        self.parser.add_argument(
            "-s", "--single", action="store_true", help="只截取当前单帧"
        )
        self.parser.add_argument(
            "-f", "--fps", type=float, default=3, help="每秒截图帧数（默认: 3）"
        )
        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=60,
            help=f"连续截图持续时间，单位秒（默认: 60，设为 0 则持续截屏直到按 '{STOP_CAPTURE_KEY}' 键停止，最大 10 分钟）",
        )
        self.parser.add_argument(
            "-k",
            "--keyboard-trigger",
            action="store_true",
            help="仅在检测到键盘输入时截图，帧保存到 actions 目录",
        )
        self.parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help=f"交互式控制：按 '{START_CAPTURE_KEY}' 开始截图，按 '{STOP_CAPTURE_KEY}' 停止截图（可与 -k 组合使用）",
        )
        self.parser.add_argument(
            "-m",
            "--minimap",
            action="store_true",
            help="仅截取小地图区域",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数。"""
        return self.parser.parse_args()


def main():
    """命令行入口。"""
    args = ScreenCapturerArgParser().parse()

    # 单帧模式不使用缓存
    use_cache = not args.single

    # 根据模式创建对应的截图器
    if args.keyboard_trigger:
        capturer = KeyboardActionCapturer(
            fps=args.fps, use_cache=use_cache, minimap_only=args.minimap
        )
    else:
        capturer = ScreenCapturer(
            fps=args.fps, use_cache=use_cache, minimap_only=args.minimap
        )

    # 创建并运行截图器运行器
    runner = CapturerRunner(capturer, keyboard_trigger=args.keyboard_trigger)
    runner.run(
        single=args.single,
        duration=args.duration,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()

    # Case: 普通模式
    # python -m gtaz.screens

    # Case: 截取单张
    # python -m gtaz.screens -s

    # Case: 连续截取，设置FPS和时长
    # python -m gtaz.screens -f 10 -d 60

    # Case: 键盘触发模式
    # python -m gtaz.screens -k -d 60

    # Case: 仅截取小地图
    # python -m gtaz.screens -m

    # Case: 键盘触发 + 仅小地图
    # python -m gtaz.screens -k -m -f 10 -d 30

    # Case: 交互式控制模式
    # python -m gtaz.screens -i

    # Case: 交互式 + 键盘触发 + 仅小地图 + FPS + 持续截图
    # python -m gtaz.screens -i -k -m -f 10 -d 0
