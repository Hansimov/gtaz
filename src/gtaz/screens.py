"""GTAV 屏幕截取"""

import argparse
import ctypes
import json
import time
import threading
import re

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
    """键盘动作信息（仅 keyboard_trigger 模式下使用）"""
    frame_index: int = 0
    """帧序号"""


class CaptureCacher:
    """
    截图缓存管理器。

    将截图数据缓存在内存中，在时间窗口结束时批量保存到文件。
    支持普通截图和带键盘动作信息的截图。
    """

    def __init__(
        self,
        save_dir: Path,
        image_format: str = IMAGE_FORMAT_JPEG,
        quality: int = DEFAULT_QUALITY,
    ):
        """
        初始化缓存管理器。

        :param save_dir: 保存目录
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"
        :param quality: JPEG 质量（1-100），默认 85
        """
        self.save_dir = save_dir
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
        filepath = self.save_dir / frame.filename

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
                    "key": key_state.key,
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

        self.save_dir.mkdir(parents=True, exist_ok=True)

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
            f"CaptureCacher("
            f"frames={len(self)}, "
            f"format={self.image_format}, "
            f"save_dir={self.save_dir})"
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
        minimap_only: bool = False,
        keyboard_trigger: bool = False,
        game_keys_only: bool = True,
    ):
        """
        初始化屏幕截取器。

        :param interval: 截图间隔时间（秒），优先级高于 fps
        :param fps: 每秒截图帧数，当 interval 未指定时使用
        :param output_dir: 输出目录，默认为 cache/frames（keyboard_trigger 时为 cache/actions）
        :param window_locator: 窗口定位器，默认为 None（将自动创建）
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"，默认 JPEG（更快更小）
        :param quality: JPEG 质量（1-100），默认 85
        :param minimap_only: 是否仅截取小地图区域，默认 False
        :param keyboard_trigger: 是否启用键盘触发模式（仅在有按键时截图），默认 False
        :param game_keys_only: 键盘触发模式下，是否只监控游戏常用按键，默认 True
        """
        self.interval = self._calculate_interval(interval, fps)
        self.window_locator = window_locator or GTAVWindowLocator()
        self.image_format = image_format.upper()
        self.quality = max(1, min(100, quality))
        self.minimap_only = minimap_only
        self.keyboard_trigger = keyboard_trigger

        # 小地图裁剪区域（首次截图时计算）
        self._minimap_crop_region: Optional[tuple[int, int, int, int]] = None

        # 键盘检测器（仅在 keyboard_trigger 模式下启用）
        self.keyboard_detector: Optional[KeyboardActionDetector] = None
        if keyboard_trigger:
            self.keyboard_detector = KeyboardActionDetector(
                game_keys_only=game_keys_only
            )

        # 生成基于启动时间的会话目录
        session_name = get_now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_dir is None:
            # 键盘触发模式使用 actions 目录，否则使用 frames 目录
            default_dir = ACTIONS_DIR if keyboard_trigger else FRAMES_DIR
            save_dir = default_dir / session_name
        else:
            save_dir = output_dir / session_name

        # 初始化缓存管理器（minimap_crop_region 在首次截图时设置）
        self.cacher = CaptureCacher(
            save_dir=save_dir,
            image_format=self.image_format,
            quality=self.quality,
        )

        # 帧计数器
        self._frame_count = 0

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
        if self.image_format == IMAGE_FORMAT_JPEG:
            ext = "jpg"
        else:
            ext = "png"
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

        self.cacher.add_frame(frame)
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

    def capture_frame(
        self,
        verbose: bool = True,
        action_info: Optional[KeyboardActionInfo] = None,
    ) -> Optional[str]:
        """
        截取当前 GTAV 窗口画面。

        :param verbose: 是否打印保存日志
        :param action_info: 键盘动作信息（可选，keyboard_trigger 模式下由 try_capture_frame 传入）
        :return: 预生成的文件名，失败则返回 None
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

        # 缓存帧数据（带或不带 action_info）
        filename = self._cache_frame(raw_data, frame_width, frame_height, action_info)

        if verbose:
            cached_count = self.get_cached_frame_count()
            if action_info:
                keys_str = ", ".join(action_info.pressed_keys)
                logger.okay(f"已截取并缓存 {cached_count} 帧 (按键: {keys_str})")
            else:
                logger.okay(f"已截取并缓存 {cached_count} 帧")
        return filename

    def try_capture_frame(self, verbose: bool = False) -> tuple[Optional[str], str]:
        """
        尝试截取一帧（用于外部循环调用）。

        普通模式：直接截图
        键盘触发模式：仅在有按键动作时截图，并记录按键信息到 action_info

        :param verbose: 是否打印日志
        :return: (文件名, 额外信息) - 如果未截图则返回 (None, "")
        """
        # 键盘触发模式：检测按键状态
        if self.keyboard_trigger and self.keyboard_detector:
            action_info = self.keyboard_detector.detect()
            if action_info.has_action:
                filename = self.capture_frame(verbose=verbose, action_info=action_info)
                extra_info = f" (按键: {', '.join(action_info.pressed_keys)})"
                return filename, extra_info
            return None, ""

        # 普通模式：直接截图
        filename = self.capture_frame(verbose=verbose)
        return filename, ""

    def flush_cache(self, verbose: bool = True) -> int:
        """
        将缓存中的所有帧保存到文件。

        :param verbose: 是否打印保存日志
        :return: 成功保存的帧数
        """
        saved_count = self.cacher.flush(verbose=verbose)
        # 保存完成后重置帧计数
        self._frame_count = 0
        return saved_count

    def get_cached_frame_count(self) -> int:
        """
        获取缓存中的帧数。

        :return: 帧数
        """
        return self.cacher.get_frame_count()

    def __del__(self):
        """析构函数，确保释放 GDI 资源。"""
        self._release_gdi_resources()

    def __repr__(self) -> str:
        parts = [
            f"ScreenCapturer(",
            f"interval={self.interval}, ",
            f"format={self.image_format}, ",
            f"quality={self.quality}, ",
            f"output_dir={self.cacher.save_dir}, ",
            f"minimap_only={self.minimap_only}, ",
        ]
        if self.keyboard_trigger:
            parts.append(f"keyboard_trigger=True, ")
        parts.extend(
            [
                f"cached_frames={len(self.cacher)}, ",
                f"frame_count={self._frame_count})",
            ]
        )
        return "".join(parts)


class CaptureLooper:
    """
    截图循环控制器。

    负责时间控制、循环逻辑、按键检测等，与具体的截图实现解耦。
    """

    def __init__(
        self,
        capturer: ScreenCapturer,
        interval: float,
    ):
        """
        初始化循环控制器。

        :param capturer: 截图器实例
        :param interval: 截图间隔（秒）
        """
        self.capturer = capturer
        self.interval = interval

        # 定时器（用于动态时间补偿）
        self._next_tick_time: float = 0.0

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

    def run_loop(
        self,
        duration: float,
        single: bool = False,
        exit_after_capture: bool = False,
        monitored_keys: Optional[list[str]] = None,
        stop_key_enabled: bool = False,
    ):
        """
        运行截图循环。

        :param duration: 持续时间（秒）
        :param single: 是否为单帧模式
        :param exit_after_capture: 截图后是否退出
        :param monitored_keys: 监控按键列表
        :param stop_key_enabled: 是否启用停止键检测
        """
        # 创建监控按键检测器
        trigger_detector = None
        last_pressed: set[str] = set()
        if monitored_keys:
            trigger_detector = KeyboardActionDetector(monitored_keys=monitored_keys)
            key_hint_str = logstr.hint(f"'{', '.join(monitored_keys)}'")
            logger.note(f"监控按键: {key_hint_str}，仅当这些键按下时才截图")

        # 创建停止键检测器
        stop_detector = None
        last_stop_key_pressed = False
        if stop_key_enabled:
            stop_detector = KeyboardActionDetector(monitored_keys=[STOP_CAPTURE_KEY])

        # 开始截图循环
        start_time = time.time()
        self.reset_tick()
        captured_count = 0

        while True:
            elapsed = time.time() - start_time
            if not single and elapsed >= duration:
                break

            # 检查停止键
            if stop_detector:
                action_info = stop_detector.detect()
                stop_key_pressed = STOP_CAPTURE_KEY in action_info.pressed_keys
                if stop_key_pressed and not last_stop_key_pressed:
                    logger.note(f"检测到 '{STOP_CAPTURE_KEY}' 键，停止截图...")
                    break
                last_stop_key_pressed = stop_key_pressed

            # 检查监控按键（边沿检测）
            should_capture = True
            if trigger_detector:
                pressed_now = set(trigger_detector.get_pressed_keys())
                newly_pressed = pressed_now - last_pressed
                last_pressed = pressed_now
                should_capture = bool(newly_pressed)

            # 执行截图
            if should_capture:
                filename, extra_info = self.capturer.try_capture_frame(verbose=False)

                # 记录进度
                if filename:
                    captured_count += 1
                    cached_count = self.capturer.get_cached_frame_count()

                    if single:
                        self.capturer.flush_cache(verbose=True)
                        logger.okay(f"单次截图成功: {filename}")
                    else:
                        # 进度日志
                        percent = (elapsed / duration) * 100
                        progress_logstr = get_progress_logstr(percent)
                        progress_str = progress_logstr(
                            f"({percent:5.1f}%) [{elapsed:.1f}/{duration:.1f}]"
                        )
                        logger.okay(
                            f"{progress_str} 已缓存 {cached_count} 帧{extra_info}"
                        )

                    # 检查是否需要退出
                    if exit_after_capture:
                        if not single:
                            logger.note(f"已截图 {captured_count} 次，退出循环")
                        break

            self.wait_next_tick()

        # 完成并保存（单帧模式已经保存过了）
        if not single and captured_count > 0:
            frame_count = self.capturer.get_cached_frame_count()
            logger.note(f"截图完成，共截取 {frame_count} 帧，开始保存...")
            saved_count = self.capturer.flush_cache(verbose=True)
            logger.okay(f"截图完成，共保存 {saved_count} 帧")


class CapturerRunner:
    """
    截图器运行器。

    封装各种运行模式的逻辑，包括单帧、连续截图、热键启停等。
    """

    def __init__(
        self,
        capturer: ScreenCapturer,
        keyboard_trigger: bool = False,
    ):
        """
        初始化截图器运行器。

        :param capturer: 截图器实例
        :param keyboard_trigger: 是否为键盘触发模式（应与 capturer.keyboard_trigger 一致）
        """
        self.capturer = capturer
        self.keyboard_trigger = keyboard_trigger

    def _print_config(self, hotkey_toggle: bool = False):
        """
        打印配置信息。

        :param hotkey_toggle: 是否为热键启停模式
        """

        if hotkey_toggle:
            hotkey_toggle_str = "热键启停"
        else:
            hotkey_toggle_str = ""

        if self.capturer.minimap_only:
            minimap_str = "（仅小地图）"
        else:
            minimap_str = ""

        if self.keyboard_trigger:
            mode_str = "键盘触发模式，"
        else:
            mode_str = ""

        logger.note(
            f"{hotkey_toggle_str}{mode_str}fps={1/self.capturer.interval:.1f}，"
            f"interval={round(self.capturer.interval, 2)}s，"
            f"cache_dir={self.capturer.cacher.save_dir}{minimap_str}"
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

    def _wait_for_start_signal(self) -> bool:
        """
        等待热键启动信号。

        :return: 是否收到启动信号（False 表示用户中断）
        """
        logger.note("热键启停模式已启动")
        logger.note(
            f"按 {key_hint(START_CAPTURE_KEY)} {val_mesg('开始截图')}，"
            f"按 {key_hint('Ctrl+C')} {val_mesg('退出')}"
        )

        hotkey_detector = KeyboardActionDetector(monitored_keys=[START_CAPTURE_KEY])
        last_start_key_pressed = False

        try:
            while True:
                action_info = hotkey_detector.detect()
                start_key_pressed = START_CAPTURE_KEY in action_info.pressed_keys

                # 边沿检测：按键从未按下变为按下（上升沿）
                if start_key_pressed and not last_start_key_pressed:
                    logger.okay(f"检测到 '{START_CAPTURE_KEY}' 键，开始截图...")
                    return True

                last_start_key_pressed = start_key_pressed
                time.sleep(0.05)  # 50ms 检测间隔

        except KeyboardInterrupt:
            logger.note("\n检测到 Ctrl+C，退出...")
            return False

    def _setup_capture_loop(
        self, duration: float, single: bool, hotkey_toggle: bool
    ) -> tuple[float, bool, Optional[KeyboardActionDetector]]:
        """
        设置截图循环的参数。

        :param duration: 持续时间（秒）
        :param single: 是否为单帧模式
        :param hotkey_toggle: 是否为热键启停模式
        :return: (实际持续时间, 是否使用停止键, 停止键检测器)
        """
        max_duration = 600  # 10分钟

        if single:
            # 单帧模式：无时间限制，只等待触发
            mode_desc = "单帧模式"
            logger.note(f"{mode_desc}，等待触发...")
            # 单帧模式下，如果是热键启停，需要监听停止键
            if hotkey_toggle:
                stop_detector = KeyboardActionDetector(
                    monitored_keys=[STOP_CAPTURE_KEY]
                )
                return max_duration, True, stop_detector
            return max_duration, False, None

        # 连续模式
        if self.keyboard_trigger:
            mode_desc = "键盘触发截图"
        else:
            mode_desc = "连续截图"

        if duration == 0 or hotkey_toggle:
            # 持续模式 或 热键启停模式：需要监听停止键
            duration = max_duration
            logger.note(
                f"{mode_desc}（持续模式，最大 {max_duration} 秒，"
                f"按 {key_hint(STOP_CAPTURE_KEY)} {val_mesg('停止截图')}），缓存模式..."
            )
            stop_detector = KeyboardActionDetector(monitored_keys=[STOP_CAPTURE_KEY])
            return duration, True, stop_detector
        else:
            # 定时模式
            logger.note(f"{mode_desc}（{duration} 秒），缓存模式...")
            return duration, False, None

    def run_capture_loop(
        self,
        duration: float = 60,
        single: bool = False,
        exit_after_capture: bool = False,
        monitored_keys: Optional[list[str]] = None,
        hotkey_toggle: bool = False,
    ):
        """
        运行截图循环（统一处理单帧和连续模式）。

        :param duration: 持续时间（秒），0表示持续模式（最大10分钟），单帧模式下忽略
        :param single: 是否为单帧模式（True=单帧，False=连续）
        :param exit_after_capture: 截图后是否退出（True=截图一次即退出，False=持续截图）
        :param monitored_keys: 监控按键名列表，仅当这些键按下时才截图（可选）
        :param hotkey_toggle: 是否为热键启停模式
        """
        # 单帧模式下，exit_after_capture 强制为 True（除非用户明确设置为 False）
        if single and exit_after_capture is None:
            exit_after_capture = True

        # 设置参数
        duration, use_stop_key, stop_detector = self._setup_capture_loop(
            duration, single, hotkey_toggle
        )

        # 创建循环控制器
        looper = CaptureLooper(
            capturer=self.capturer,
            interval=self.interval,
        )

        # 执行截图循环
        looper.run_loop(
            duration=duration,
            single=single,
            exit_after_capture=exit_after_capture,
            monitored_keys=monitored_keys,
            stop_key_enabled=use_stop_key,
        )

    def run(
        self,
        single: bool = False,
        duration: float = 60,
        hotkey_toggle: bool = False,
        exit_after_capture: bool = True,
        monitored_keys: Optional[list[str]] = None,
    ):
        """
        运行截图器。

        :param single: 是否只截取单帧
        :param duration: 连续截图持续时间（秒）
        :param hotkey_toggle: 是否为热键启停模式
        :param exit_after_capture: 截图后是否退出（单帧模式默认 True，连续模式默认 False）
        :param monitored_keys: 监控按键名列表，仅当这些键按下时才截图
        """
        # 打印配置信息
        self._print_config(hotkey_toggle=hotkey_toggle)

        # 验证窗口
        if not self._validate_window():
            return

        logger.note(f"截取器信息: {self.capturer}")

        # 热键启停模式：等待启动信号
        if hotkey_toggle:
            if not self._wait_for_start_signal():
                return  # 用户中断，退出

        # 统一使用 run_capture_loop，通过参数区分单帧/连续模式
        # 连续模式下，exit_after_capture 默认为 False
        if not single:
            exit_after_capture = False

        try:
            self.run_capture_loop(
                duration=duration,
                single=single,
                exit_after_capture=exit_after_capture,
                monitored_keys=monitored_keys,
                hotkey_toggle=hotkey_toggle,
            )
        except KeyboardInterrupt:
            logger.note("\n检测到 Ctrl+C，正在退出...")
            # 如果有缓存的帧，保存它们
            if self.capturer.get_cached_frame_count() > 0:
                self.capturer.flush_cache(verbose=True)


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
            "-x",
            "--exit-after-capture",
            action="store_true",
            default=False,
            help="单帧模式下，截图后退出（默认不退出，继续监听触发事件）",
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
            "-t",
            "--hotkey-toggle",
            action="store_true",
            dest="hotkey_toggle",
            help=f"热键启停，按 '{START_CAPTURE_KEY}' 开始截图，按 '{STOP_CAPTURE_KEY}' 停止截图（可与 -k 组合使用）",
        )
        self.parser.add_argument(
            "-k",
            "--keyboard-trigger",
            action="store_true",
            help="仅在有键盘输入时截图，并记录按键信息",
        )
        self.parser.add_argument(
            "-r",
            "--monitored-keys",
            type=str,
            default="",
            help="单帧模式下，按下指定键才触发截图，按住不重复触发",
        )
        self.parser.add_argument(
            "-m",
            "--minimap-only",
            action="store_true",
            help="仅截取小地图区域",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数。"""
        return self.parser.parse_args()


def main():
    """命令行入口。"""
    args = ScreenCapturerArgParser().parse()

    # 创建统一的截图器，通过 keyboard_trigger 参数控制模式
    capturer = ScreenCapturer(
        fps=args.fps,
        minimap_only=args.minimap_only,
        keyboard_trigger=args.keyboard_trigger,
    )

    # 解析监控按键
    monitored_keys = None
    if args.monitored_keys:
        monitored_keys = [
            k.strip() for k in args.monitored_keys.split(",") if k.strip()
        ]

    # 创建并运行截图器运行器
    runner = CapturerRunner(capturer, keyboard_trigger=args.keyboard_trigger)
    runner.run(
        single=args.single,
        duration=args.duration,
        hotkey_toggle=args.hotkey_toggle,
        exit_after_capture=args.exit_after_capture,
        monitored_keys=monitored_keys,
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

    # Case: 热键启停模式
    # python -m gtaz.screens -t

    # Case: 热键启停 + 键盘触发 + 仅小地图 + FPS + 持续截图
    # python -m gtaz.screens -t -k -m -f 10 -d 0
