"""GTAV 屏幕截取"""

import argparse
import ctypes
import json
import time
import threading

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image
from tclogger import PathType, TCLogger, TCLogbar, logstr


from .windows import GTAVWindowLocator
from .keyboard_actions import KeyboardActionDetector, KeyboardActionInfo
from .keyboard_actions import TriggerType, KEY_UP, KEY_DOWN, KEY_HOLD
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

# 默认截图帧率
FPS = 10

# 默认图像格式
IMAGE_FORMAT = "jpeg"

# 默认图像质量（JPEG）
DEFAULT_QUALITY = 85

# 录制启停热键
START_RECORD_KEY = "1"
STOP_RECORD_KEY = "2"

# 最大录制时长（秒）
MAX_DURATION = 600


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


def is_jpeg(image_format: str) -> bool:
    """检查图像格式是否为 JPEG。"""
    fmt = image_format.lstrip(".").lower()
    return fmt == "jpeg" or fmt == "jpg"


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
    action_info: KeyboardActionInfo = None
    """键盘动作信息"""
    frame_index: int = 0
    """帧序号"""


class DetectorManager:
    """创建和管理不同用途的键盘输入检测器。"""

    @staticmethod
    def create_capture_detector(
        monitor_keys: list[str] = None,
        trigger_type: TriggerType = None,
    ) -> KeyboardActionDetector:
        """创建截图触发检测器

        :param monitor_keys: 监控按键列表，默认为 None（使用游戏常用按键）
        :param trigger_type: 按键触发类型
        :param exclude_keys: 要排除的按键列表，默认排除控制键

        :return: 键盘检测器实例
        """
        # 默认排除启停控制键
        exclude_keys = [START_RECORD_KEY, STOP_RECORD_KEY]
        if monitor_keys:
            return KeyboardActionDetector(
                monitor_keys=monitor_keys,
                trigger_type=trigger_type,
                exclude_keys=exclude_keys,
            )
        else:
            return KeyboardActionDetector(
                game_keys_only=True,
                trigger_type=trigger_type,
                exclude_keys=exclude_keys,
            )

    @staticmethod
    def create_start_detector() -> KeyboardActionDetector:
        """创建启动热键检测器

        :return: 键盘检测器实例
        """
        return KeyboardActionDetector(
            monitor_keys=[START_RECORD_KEY], trigger_type=KEY_DOWN
        )

    @staticmethod
    def create_stop_detector() -> KeyboardActionDetector:
        """创建停止热键检测器

        :return: 键盘检测器实例
        """
        return KeyboardActionDetector(
            monitor_keys=[STOP_RECORD_KEY], trigger_type=KEY_DOWN
        )


class CaptureCacher:
    """
    截图缓存管理器。

    将截图数据缓存在内存中，在时间窗口结束时批量保存到文件。可以保存图片和键盘动作信息。
    """

    def __init__(
        self,
        save_dir: Path,
        image_format: str = IMAGE_FORMAT,
        quality: int = DEFAULT_QUALITY,
    ):
        """
        初始化缓存管理器。

        :param save_dir: 保存目录
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"
        :param quality: JPEG 质量（1-100），默认 85
        """
        self.save_dir = save_dir
        self.image_format = image_format
        self.quality = max(1, min(100, quality))

        self._frames: list[CapturedFrame] = []
        self._frame_count: int = 0
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

        image_path = self.save_dir / frame.filename

        # 根据格式保存
        if is_jpeg(self.image_format):
            image = image.convert("RGB")
            image.save(image_path, "JPEG", quality=self.quality, optimize=False)
        else:
            image = image.convert("RGB")
            image.save(image_path, "PNG", compress_level=1)

        return image_path

    def _save_action_info(self, frame: CapturedFrame, image_path: Path) -> bool:
        """
        保存键盘动作信息到 JSON 文件。

        :param frame: 截图帧数据
        :param image_path: 图像文件路径

        :return: 是否保存成功
        """
        if not frame.action_info:
            return False
        json_path = image_path.with_suffix(".json")

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
                "image": image_path.name,
                "json": json_path.name,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    def flush(self, verbose: bool = True) -> int:
        """
        将缓存中的所有帧保存到文件。

        :param verbose: 是否打印保存日志
        :return: 成功保存的帧数
        """
        with self._lock:
            frames = self._frames.copy()

        if not frames:
            return 0

        if verbose:
            logger.note(f"开始保存 {len(frames)} 帧到文件...")
            bar = TCLogbar(total=len(frames), desc="* 保存截图")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for i, frame in enumerate(frames):
            # 保存图像
            image_path = self._save_single_image(frame)
            if not image_path:
                continue
            saved_count += 1
            # 保存键盘动作
            if frame.action_info:
                self._save_action_info(frame=frame, image_path=image_path)
            if verbose:
                bar.update(1)

        if verbose:
            bar.update(flush=True)
            print()
            logger.okay(f"保存完成，共 {saved_count}/{len(frames)} 帧")

        with self._lock:
            self._frames.clear()
            self._frame_count = 0

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
    窗口画面截取器。

    按照指定的时间间隔截取窗口画面，并保存到本地文件。支持后台截取。
    """

    def __init__(
        self,
        interval: float = None,
        fps: float = None,
        output_dir: PathType = None,
        window_locator: GTAVWindowLocator = None,
        image_format: str = IMAGE_FORMAT,
        quality: int = DEFAULT_QUALITY,
        minimap_only: bool = False,
        capture_detector: KeyboardActionDetector = None,
    ):
        """
        初始化屏幕截取器。

        :param interval: 截图间隔时间（秒），优先级高于 fps
        :param fps: 每秒截图帧数，当 interval 未指定时使用
        :param output_dir: 输出目录，默认根据 capture_detector 决定
        :param window_locator: 窗口定位器，默认为 None（将自动创建）
        :param image_format: 图像格式，支持 "JPEG" 或 "PNG"，默认 JPEG（更快更小）
        :param quality: JPEG 质量（1-100），默认 85
        :param minimap_only: 是否仅截取小地图区域，默认 False
        :param capture_detector: 截图触发检测器，None 表示按间隔截图，非 None 表示键盘触发模式（仅在有按键时截图）
        """
        self._init_fps_interval(interval, fps)
        self.window_locator = window_locator or GTAVWindowLocator()
        self.image_format = image_format
        self.quality = max(1, min(100, quality))
        self.minimap_only = minimap_only
        self.capture_detector = capture_detector

        # 小地图裁剪区域（首次截图时计算）
        self._minimap_crop_region: tuple[int, int, int, int] = None

        # 生成基于启动时间的会话目录
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_dir is None:
            # 有 capture_detector 使用 actions 目录，否则使用 frames 目录
            if capture_detector:
                output_dir = ACTIONS_DIR
            else:
                output_dir = FRAMES_DIR
        save_dir = Path(output_dir) / session_name

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

    def _init_fps_interval(self, interval: float = None, fps: float = None):
        """
        计算截图间隔时间。优先级: interval > fps > 默认值

        :param interval: 截图间隔时间（秒）
        :param fps: 每秒截图帧数
        """
        if interval is not None and interval > 0:
            self.interval = interval
            self.fps = 1.0 / interval
        elif fps is not None and fps > 0:
            self.fps = fps
            self.interval = 1.0 / fps
        else:
            self.fps = None
            self.interval = None

    def _generate_filename(self, frame_index: int) -> str:
        """
        生成截图文件名，格式为 YYYY-MM-DD_HH-MM-SS-sss_<frame_idx>.ext

        :param frame_index: 帧索引（0000-9999）
        :return: 文件名字符串
        """
        now = datetime.now()
        if is_jpeg(self.image_format):
            ext = "jpg"
        else:
            ext = "png"
        return (
            now.strftime("%Y-%m-%d_%H-%M-%S-")
            + f"{now.microsecond // 1000:03d}_{frame_index:04d}.{ext}"
        )

    def _get_window_info(self) -> tuple[int, int, int]:
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

    def _capture_window(self, hwnd: int, width: int, height: int) -> bytes:
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
        action_info: KeyboardActionInfo = None,
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
        self, action_info: KeyboardActionInfo = None, verbose: bool = True
    ) -> str:
        """
        截取当前 GTAV 窗口画面。

        :param verbose: 是否打印保存日志
        :param action_info: 键盘动作信息

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

        # 缓存帧数据
        filename = self._cache_frame(raw_data, frame_width, frame_height, action_info)

        if verbose:
            cached_count = self.get_cached_frame_count()
            if action_info:
                keys_str = ", ".join(action_info.pressed_keys)
                logger.okay(f"已截取并缓存 {cached_count} 帧 (按键: {keys_str})")
            else:
                logger.okay(f"已截取并缓存 {cached_count} 帧")
        return filename

    def try_capture_frame(self, verbose: bool = False) -> tuple[str, str]:
        """
        尝试截取一帧（用于外部循环调用）。

        普通模式：直接截图
        键盘触发模式：仅在有按键动作时截图

        :param verbose: 是否打印日志
        :return: (文件名, 额外信息) - 如果未截图则返回 (None, "")
        """
        # 键盘触发模式：检测按键状态
        if self.capture_detector:
            action_info = self.capture_detector.detect()
            if action_info.has_action:
                # 有 capture_detector 则保存按键详细信息
                filename = self.capture_frame(action_info=action_info, verbose=verbose)
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
        if self.fps:
            fps_str = f"{self.fps:.1f}"
        else:
            fps_str = "None"

        if self.interval:
            interval_str = f"{round(self.interval, 2)}s"
        else:
            interval_str = "None"

        if self.capture_detector:
            detector_str = "True"
        else:
            detector_str = "None"

        parts = [
            f"ScreenCapturer(",
            f"fps={fps_str}, ",
            f"interval={interval_str}, ",
            f"image_format={self.image_format}, ",
            f"quality={self.quality}, ",
            f"save_dir={self.cacher.save_dir}, ",
            f"minimap_only={self.minimap_only}, ",
            f"capture_detector={detector_str}",
            f"cached_frames={len(self.cacher)}, ",
            f"frame_count={self._frame_count})",
        ]

        return "".join(parts)


class RecordRunner:
    def __init__(
        self,
        capturer: ScreenCapturer,
        duration: float = None,
        single: bool = False,
        hotkey_toggle: bool = False,
        exit_after_record: bool = False,
    ):
        """
        初始化录制运行器。

        :param capturer: 截图器实例
        :param duration: 持续时间（秒），0表示持续模式
        :param single: 是否为单帧模式
        :param hotkey_toggle: 是否启用热键启停模式
        :param exit_after_record: 录制后是否退出（默认 False，继续监听新的录制触发事件）
        """
        self.capturer = capturer
        self.duration = duration
        self.single = single
        self.hotkey_toggle = hotkey_toggle
        self.exit_after_record = exit_after_record
        self._create_detectors()

    def _create_detectors(self):
        """根据启停模式创建检测器"""
        if self.hotkey_toggle:
            self.start_detector = DetectorManager.create_start_detector()
            self.stop_detector = DetectorManager.create_stop_detector()
        else:
            self.start_detector = None
            self.stop_detector = None

    def _check_window(self) -> bool:
        """检测窗口"""
        if not self.capturer.window_locator.is_window_valid():
            erro_str = f"窗口 [{self.capturer.window_locator.window_title}] 未找到"
            logger.erro(erro_str)
            raise RuntimeError(erro_str)
        return True

    def _wait_until_next_tick(
        self, next_tick_time: float, frame_captured: bool
    ) -> float:
        """
        等待直到下一次 tick。

        :param next_tick_time: 下一次tick的时间戳
        :param frame_captured: 本次是否截取了帧

        :return: 更新后的下一次tick时间戳
        """
        # 键盘触发模式下，且未截图：快速检测按键（5ms）
        if self.capturer.capture_detector and not frame_captured:
            time.sleep(0.005)
            return next_tick_time

        # 其他情况：按帧率间隔等待
        next_tick_time += self.capturer.interval
        sleep_time = next_tick_time - time.time()

        # 睡眠至下一次 tick
        if sleep_time > 0:
            time.sleep(sleep_time)
        # 如果已超时，重置为当前时间
        else:
            next_tick_time = time.time()

        return next_tick_time

    def _log_keyboard_interrupt(self):
        logger.note(f"\n检测到 {key_hint('Ctrl+C')}，正在退出...")

    def _wait_start_signal(self) -> bool:
        """
        等待热键启动信号。

        :return: 是否收到启动信号（False 表示用户中断）
        :raises KeyboardInterrupt: 当用户按下 Ctrl+C 时
        """
        logger.note("热键启停模式已启动")
        logger.note(
            f"按 {key_hint(START_RECORD_KEY)} {val_mesg('开始录制')}，"
            f"按 {key_hint(STOP_RECORD_KEY)} {val_mesg('停止录制')}，"
            f"按 {key_hint('Ctrl+C')} {val_mesg('退出')}"
        )

        while True:
            action_info = self.start_detector.detect()
            if action_info.has_action:
                logger.okay(f"检测到 {key_hint(START_RECORD_KEY)} 键，开始录制...")
                return True
            time.sleep(0.015)  # 15ms/tick

    def _log_loop_progress(self, elapsed: float, extra_info: str = ""):
        """循环进度日志"""
        cached_count = self.capturer.get_cached_frame_count()
        percent = (elapsed / self.duration) * 100
        progress_logstr = get_progress_logstr(percent)
        progress_str = progress_logstr(
            f"({percent:5.1f}%) [{elapsed:.1f}/{self.duration:.1f}]"
        )
        logger.okay(f"{progress_str} 已缓存 {cached_count} 帧{extra_info}")

    def _log_duration(self):
        if self.single:
            # 单帧模式，duration 不影响
            logger.note("单帧模式：等待触发...")
        elif self.duration == 0:
            # 持续模式，duration 最大10分钟
            self.duration = MAX_DURATION
            logger.note(
                f"持续模式：按 {key_hint(STOP_RECORD_KEY)} {val_mesg('停止录制')}..."
            )
        else:
            logger.note(f"定时模式：{self.duration} 秒...")

    def _save_captured_frames(self):
        """保存已截取的帧"""
        frame_count = self.capturer.get_cached_frame_count()
        logger.note(f"截取完成，共截取 {frame_count} 帧，开始保存...")
        saved_count = self.capturer.flush_cache(verbose=True)
        logger.okay(f"保存完成，共保存 {saved_count} 帧")

    def _finalize(self):
        if self.capturer.get_cached_frame_count() > 0:
            self._save_captured_frames()

    def _run_loop(self):
        """运行截图循环"""
        # 热键启停模式：等待启动信号
        if self.start_detector:
            self._wait_start_signal()

        # 持续时间日志
        self._log_duration()

        if self.capturer.capture_detector:
            # 键盘触发模式：start_time 在第一次截图时初始化
            start_time = None
        else:
            # 普通模式：立即开始计时
            start_time = time.time()
        next_tick_time = time.time()
        captured_count = 0

        # 主循环
        while True:
            # 键盘触发模式下，首次截图前始终重置 elapsed = 0
            if start_time is None:
                elapsed = 0
            else:
                elapsed = time.time() - start_time

            if not self.single and elapsed >= self.duration:
                break

            # 检查停止键
            if self.stop_detector:
                action_info = self.stop_detector.detect()
                if action_info.has_action:
                    logger.note(f"检测到 {key_hint(STOP_RECORD_KEY)} 键，停止录制...")
                    break

            # 运行截图
            filename, extra_info = self.capturer.try_capture_frame(verbose=False)
            if filename:
                captured_count += 1

                # 键盘触发模式：首次截图时初始化 start_time
                if start_time is None:
                    start_time = time.time()
                    next_tick_time = start_time
                    elapsed = 0  # 首次截图，elapsed 为 0

                if self.single:
                    self.capturer.flush_cache(verbose=True)
                    logger.okay(f"单帧截图成功: {filename}")
                    break

                self._log_loop_progress(elapsed, extra_info)

            # 等待至下一次 tick
            next_tick_time = self._wait_until_next_tick(next_tick_time, bool(filename))

        # 完成并保存
        if not self.single and captured_count > 0:
            self._save_captured_frames()

    def run(self):
        """运行截图器"""
        self._check_window()
        logger.note(f"截取器信息: {self.capturer}")

        try:
            # 执行截图循环
            if self.exit_after_record:
                self._run_loop()
            else:
                while True:
                    self._run_loop()
        except KeyboardInterrupt:
            self._log_keyboard_interrupt()
            self._finalize()


class ScreenCapturerArgParser:
    """屏幕截取器命令行参数解析器。"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GTAV 屏幕截取器")
        self._add_arguments()

    def _add_arguments(self):
        """添加命令行参数。"""
        self.parser.add_argument(
            "-s", "--single", action="store_true", help="单帧截取模式"
        )
        self.parser.add_argument(
            "-x",
            "--exit-after-record",
            action="store_true",
            default=False,
            help="录制后退出（默认不退出，继续监听新的录制触发事件）",
        )
        self.parser.add_argument(
            "-f", "--fps", type=float, default=None, help="每秒截图帧数"
        )
        self.parser.add_argument(
            "-o", "--output-dir", type=str, default=None, help="截图文件保存父目录"
        )
        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=None,
            help=f"连续截图持续时间，单位秒",
        )
        self.parser.add_argument(
            "-g",
            "--hotkey-toggle",
            action="store_true",
            help=f"使用热键启停录制，按 '{START_RECORD_KEY}' 开始录制，按 '{STOP_RECORD_KEY}' 停止录制（可与 -k 组合使用）",
        )
        self.parser.add_argument(
            "-i",
            "--input-trigger",
            action="store_true",
            help="仅在有键盘输入时截图，并记录按键信息",
        )
        self.parser.add_argument(
            "-k",
            "--monitor-keys",
            type=str,
            default="",
            help="只有指定的按键才能触发截图",
        )
        self.parser.add_argument(
            "-t",
            "--trigger-type",
            type=str,
            default=None,
            choices=["down", "hold"],
            help="按键触发类型（down=边沿触发/刚按下，hold=电平触发/按住）",
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

    # 解析监控按键
    if args.monitor_keys:
        monitor_keys = [k.strip() for k in args.monitor_keys.split(",") if k.strip()]
    else:
        monitor_keys = None

    # 设置触发类型
    if args.trigger_type:
        trigger_type = args.trigger_type
    else:
        if args.single:
            trigger_type = KEY_DOWN
        else:
            trigger_type = KEY_HOLD

    # 创建截图触发检测器
    if args.input_trigger or monitor_keys:
        capture_detector = DetectorManager.create_capture_detector(
            monitor_keys=monitor_keys, trigger_type=trigger_type
        )
    else:
        capture_detector = None

    # 设置 fps
    if not args.single and not args.fps:
        fps = FPS
    else:
        fps = args.fps

    # 创建截图器
    capturer = ScreenCapturer(
        fps=fps,
        output_dir=args.output_dir,
        minimap_only=args.minimap_only,
        capture_detector=capture_detector,
    )

    # 创建录制运行器，然后运行
    runner = RecordRunner(
        capturer=capturer,
        duration=args.duration,
        single=args.single,
        hotkey_toggle=args.hotkey_toggle,
        exit_after_record=args.exit_after_record,
    )
    runner.run()


if __name__ == "__main__":
    main()

    # Case: 截取单张
    # python -m gtaz.screens -s

    # Case: 单帧截取，按下特定键截取（单帧模式下，触发类型默认 KEY_DOWN：按下截图一次，不重复截取）
    # python -m gtaz.screens -s -k k

    # Case: 连续截取，设置FPS和时长
    # python -m gtaz.screens -f 10 -d 60

    # Case: 键盘触发模式（连续模式下，触发类型默认 KEY_HOLD：按住则持续截图）
    # python -m gtaz.screens -i -f 10 -d 60

    # Case: 键盘触发 + 仅小地图
    # python -m gtaz.screens -i -m -f 10 -d 30

    # Case: 热键启停 + 键盘触发 + 单帧
    # python -m gtaz.screens -g -i -s

    # Case: 热键启停 + 键盘触发 + 仅小地图 + FPS + 持续截图
    # python -m gtaz.screens -g -i -m -f 10 -d 0
