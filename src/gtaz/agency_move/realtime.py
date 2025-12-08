"""
GTAV 实时行为克隆控制模块

实时捕获 GTAV 游戏画面（小地图），使用深度学习模型进行推理，
并将预测的键盘操作（WASD）映射为手柄摇杆操作。

## 主要功能

1. 实时捕获 GTAV 窗口的小地图区域（10 FPS）
2. 支持多种推理后端：TensorRT > ONNX Runtime > PyTorch（按优先级自动选择）
3. 将 WASD 键盘预测映射为手柄左摇杆操作
4. 支持交互式控制（按键启动/停止）

## 使用示例

```sh
# 使用默认模型运行（自动选择最佳后端）
python -m gtaz.agency_move.realtime

# 指定模型路径（支持 .engine, .onnx, .pth）
python -m gtaz.agency_move.realtime -m path/to/model.engine
python -m gtaz.agency_move.realtime -m path/to/model.onnx
python -m gtaz.agency_move.realtime -m path/to/model.pth

# 设置帧率和推理阈值
python -m gtaz.agency_move.realtime -f 10 -t 0.5
```

## 推理后端优先级

1. TensorRT (.engine) - 最快，约 2ms 延迟
2. ONNX Runtime (.onnx) - 较快，约 5ms 延迟
3. PyTorch (.pth) - 最慢但最通用，约 5ms 延迟

## WASD 到摇杆映射

- W: 摇杆向上
- S: 摇杆向下
- A: 摇杆向左
- D: 摇杆向右
- W+A: 摇杆左上
- W+D: 摇杆右上
- S+A: 摇杆左下
- S+D: 摇杆右下
- 无按键: 摇杆回中

## 交互式控制

- 按 '3' 键启动控制
- 按 '4' 键停止控制
- 按 Ctrl+C 退出程序
"""

import argparse
import ctypes
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tclogger import TCLogger, logstr, Runtimer
from torchvision import transforms

# 本地模块导入
from ..windows import GTAVWindowLocator
from ..gamepads import GamepadSimulator, JoystickDirection
from ..segments import calc_minimap_crop_region

# 尝试导入推理模块
try:
    from .inference import (
        TensorRTInferencer,
        ONNXRuntimeInferencer,
        PyTorchInferencer,
        InferenceConfig,
        CKPT_DIR,
        NUM_KEYS,
        INDEX_TO_KEY,
        TENSORRT_AVAILABLE,
        ONNX_AVAILABLE,
    )
except ImportError as e:
    TENSORRT_AVAILABLE = False
    ONNX_AVAILABLE = False
    print(f"Warning: Inference module not available. ({e})")


logger = TCLogger(name="RealtimeControl", use_prefix=True, use_prefix_ms=True)


# ===== 常量定义 ===== #

# Windows API 常量
SRCCOPY = 0x00CC0020
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002

# 默认参数
DEFAULT_FPS = 10
DEFAULT_THRESHOLD = 0.5
DEFAULT_HISTORY_FRAMES = 4

# 交互式控制键
START_KEY_VK = 0x33  # '3' 键
STOP_KEY_VK = 0x34  # '4' 键


# ===== 配置类 ===== #


@dataclass
class RealtimeConfig:
    """实时控制配置"""

    # 捕获参数
    fps: float = DEFAULT_FPS
    history_frames: int = DEFAULT_HISTORY_FRAMES

    # 推理参数
    threshold: float = DEFAULT_THRESHOLD
    image_size: tuple[int, int] = (160, 220)  # (H, W)

    # 手柄参数
    joystick_strength: float = 1.0  # 摇杆推动强度 (0.0-1.0)

    # 模型路径
    model_path: Optional[str] = None  # 可以是 .engine, .onnx, 或 .pth 文件

    @property
    def interval(self) -> float:
        """计算帧间隔"""
        return 1.0 / self.fps


# ===== WASD 到摇杆映射 ===== #


class WASDToJoystickMapper:
    """WASD 键盘输入到摇杆方向的映射器"""

    def __init__(self, strength: float = 1.0):
        """
        初始化映射器。

        :param strength: 摇杆推动强度 (0.0-1.0)
        """
        self.strength = min(1.0, max(0.0, strength))

    def map(self, wasd: dict[str, bool]) -> tuple[float, float]:
        """
        将 WASD 按键状态映射为摇杆方向。

        :param wasd: WASD 按键状态字典，如 {"W": True, "A": False, "S": False, "D": True}
        :return: (x, y) 摇杆方向，x 为左右（-1到1），y 为前后（-1到1）
        """
        x = 0.0
        y = 0.0

        # 前后方向（W/S）
        if wasd.get("W", False):
            y += 1.0
        if wasd.get("S", False):
            y -= 1.0

        # 左右方向（A/D）
        if wasd.get("A", False):
            x -= 1.0
        if wasd.get("D", False):
            x += 1.0

        # 归一化对角线方向（使其长度为1）
        if x != 0 and y != 0:
            magnitude = (x**2 + y**2) ** 0.5
            x /= magnitude
            y /= magnitude

        # 应用强度
        x *= self.strength
        y *= self.strength

        return (x, y)

    def map_from_prediction(self, preds: np.ndarray) -> tuple[float, float]:
        """
        从模型预测结果映射为摇杆方向。

        :param preds: 模型预测结果，shape (4,) 对应 [W, A, S, D]
        :return: (x, y) 摇杆方向
        """
        wasd = {
            "W": bool(preds[0] > 0.5),
            "A": bool(preds[1] > 0.5),
            "S": bool(preds[2] > 0.5),
            "D": bool(preds[3] > 0.5),
        }
        return self.map(wasd)


# ===== 实时图像捕获器 ===== #


class RealtimeFrameCapturer:
    """实时帧捕获器，将截图保存在内存中而不是文件"""

    def __init__(
        self,
        window_locator: GTAVWindowLocator,
        image_size: tuple[int, int] = (160, 220),
        minimap_only: bool = True,
    ):
        """
        初始化帧捕获器。

        :param window_locator: GTAV 窗口定位器
        :param image_size: 输出图像尺寸 (H, W)
        :param minimap_only: 是否仅截取小地图区域
        """
        self.window_locator = window_locator
        self.image_size = image_size
        self.minimap_only = minimap_only

        # 小地图裁剪区域（首次截图时计算）
        self._minimap_crop_region: Optional[tuple[int, int, int, int]] = None

        # 加载 Windows API
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32

        # 缓存的 GDI 资源
        self._cached_width: int = 0
        self._cached_height: int = 0
        self._cached_dc = None
        self._cached_bitmap = None
        self._cached_bmp_info = None

        # 图像预处理 transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _create_bitmap_info(self, width: int, height: int) -> ctypes.Array:
        """创建 BITMAPINFOHEADER 结构"""
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
        """确保 GDI 资源已创建"""
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
        """释放 GDI 资源"""
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
        """在字节级别裁剪 BGRA 原始数据"""
        left, top, right, bottom = crop_region
        crop_width = right - left
        crop_height = bottom - top
        bytes_per_pixel = 4  # BGRA
        src_stride = src_width * bytes_per_pixel
        crop_stride = crop_width * bytes_per_pixel

        cropped_rows = []
        for y in range(top, bottom):
            row_start = y * src_stride + left * bytes_per_pixel
            row_end = row_start + crop_stride
            cropped_rows.append(raw_data[row_start:row_end])

        return b"".join(cropped_rows), crop_width, crop_height

    def _ensure_minimap_crop_region(self, width: int, height: int):
        """确保小地图裁剪区域已计算"""
        if self._minimap_crop_region is None:
            self._minimap_crop_region = calc_minimap_crop_region(width, height)
            logger.note(f"小地图裁剪区域: {self._minimap_crop_region}")

    def capture_raw(self) -> Optional[tuple[bytes, int, int]]:
        """
        捕获窗口原始数据。

        :return: (raw_data, width, height) 或 None
        """
        hwnd = self.window_locator.hwnd
        if not hwnd:
            return None

        client_size = self.window_locator.get_client_size()
        if not client_size:
            return None

        width, height = client_size
        if width <= 0 or height <= 0:
            return None

        # 获取窗口 DC
        hwnd_dc = self.user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            return None

        # 确保 GDI 资源已准备好
        self._ensure_gdi_resources(hwnd_dc, width, height)

        # 选择位图到 DC
        old_bitmap = self.gdi32.SelectObject(self._cached_dc, self._cached_bitmap)

        # 使用 PrintWindow 截取窗口内容
        result = self.user32.PrintWindow(
            hwnd, self._cached_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT
        )

        if not result:
            # 如果 PrintWindow 失败，尝试使用 BitBlt
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

        raw_data = buffer.raw

        # 如果需要裁剪小地图
        if self.minimap_only:
            self._ensure_minimap_crop_region(width, height)
            raw_data, width, height = self._crop_raw_data(
                raw_data, width, height, self._minimap_crop_region
            )

        return raw_data, width, height

    def capture_tensor(self) -> Optional[np.ndarray]:
        """
        捕获并预处理为 tensor。

        :return: 预处理后的 numpy array，shape (3, H, W)，或 None
        """
        result = self.capture_raw()
        if result is None:
            return None

        raw_data, width, height = result

        # 转换为 PIL Image
        image = Image.frombuffer(
            "RGBA", (width, height), raw_data, "raw", "BGRA", 0, 1
        ).convert("RGB")

        # 应用预处理
        tensor = self.transform(image)

        return tensor.numpy()

    def __del__(self):
        """清理资源"""
        self._release_gdi_resources()


# ===== 实时控制器 ===== #


class RealtimeController:
    """实时行为克隆控制器"""

    # 支持的推理后端优先级顺序
    BACKEND_PRIORITY = ["tensorrt", "onnx", "pytorch"]

    def __init__(self, config: RealtimeConfig = None):
        """
        初始化控制器。

        :param config: 配置对象
        """
        self.config = config or RealtimeConfig()

        # 初始化组件
        logger.note("初始化组件...")

        # 窗口定位器
        self.window_locator = GTAVWindowLocator()
        if not self.window_locator.is_window_valid():
            raise RuntimeError("未找到 GTAV 窗口，请确保游戏已启动")

        # 帧捕获器
        self.frame_capturer = RealtimeFrameCapturer(
            window_locator=self.window_locator,
            image_size=self.config.image_size,
            minimap_only=True,
        )

        # 创建推理配置
        infer_config = InferenceConfig(
            image_size=self.config.image_size,
            threshold=self.config.threshold,
            history_frames=self.config.history_frames,
        )

        # 初始化推理器（自动选择最佳后端）
        self.inferencer, self.backend_name = self._create_inferencer(infer_config)

        # 手柄模拟器
        self.gamepad = GamepadSimulator()

        # WASD 到摇杆映射器
        self.mapper = WASDToJoystickMapper(strength=self.config.joystick_strength)

        # 帧历史缓冲区
        self.frame_buffer: deque = deque(maxlen=self.config.history_frames)

        # 按键历史缓冲区
        self.key_history: deque = deque(maxlen=self.config.history_frames - 1)

        # 控制标志
        self._running = False
        self._control_thread: Optional[threading.Thread] = None

        # 统计信息
        self._frame_count = 0
        self._inference_times: deque = deque(maxlen=100)

        logger.okay(f"控制器初始化完成，使用后端: {self.backend_name}")

    def _create_inferencer(self, infer_config: InferenceConfig):
        """
        创建推理器，按优先级尝试不同后端。

        优先级: TensorRT (.engine) > ONNX Runtime (.onnx) > PyTorch (.pth)

        :param infer_config: 推理配置
        :return: (inferencer, backend_name)
        """
        # 如果用户指定了模型路径，根据扩展名选择后端
        if self.config.model_path:
            model_path = Path(self.config.model_path)
            suffix = model_path.suffix.lower()

            if suffix == ".engine":
                return self._try_tensorrt(str(model_path), infer_config)
            elif suffix == ".onnx":
                return self._try_onnx(str(model_path), infer_config)
            elif suffix == ".pth":
                return self._try_pytorch(str(model_path), infer_config)
            else:
                raise ValueError(f"不支持的模型格式: {suffix}")

        # 自动查找模型，按优先级尝试
        model_base = self._find_latest_model_base()

        # 尝试 TensorRT
        engine_path = model_base.with_suffix(".engine")
        if engine_path.exists() and TENSORRT_AVAILABLE:
            try:
                return self._try_tensorrt(str(engine_path), infer_config)
            except Exception as e:
                logger.warn(f"TensorRT 初始化失败: {e}")

        # 尝试 ONNX Runtime
        onnx_path = model_base.with_suffix(".onnx")
        if onnx_path.exists() and ONNX_AVAILABLE:
            try:
                return self._try_onnx(str(onnx_path), infer_config)
            except Exception as e:
                logger.warn(f"ONNX Runtime 初始化失败: {e}")

        # 尝试 PyTorch
        pth_path = model_base.with_suffix(".pth")
        if pth_path.exists():
            try:
                return self._try_pytorch(str(pth_path), infer_config)
            except Exception as e:
                logger.warn(f"PyTorch 初始化失败: {e}")

        raise RuntimeError("无法初始化任何推理后端，请检查模型文件是否存在")

    def _try_tensorrt(self, engine_path: str, infer_config: InferenceConfig):
        """尝试创建 TensorRT 推理器"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT 不可用")
        logger.note(f"尝试使用 TensorRT: {Path(engine_path).name}")
        inferencer = TensorRTInferencer(engine_path, config=infer_config)
        logger.okay(f"TensorRT 推理器初始化成功")
        return inferencer, "TensorRT"

    def _try_onnx(self, onnx_path: str, infer_config: InferenceConfig):
        """尝试创建 ONNX Runtime 推理器"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime 不可用")
        logger.note(f"尝试使用 ONNX Runtime: {Path(onnx_path).name}")
        inferencer = ONNXRuntimeInferencer(onnx_path, config=infer_config)
        logger.okay(f"ONNX Runtime 推理器初始化成功")
        return inferencer, "ONNX Runtime"

    def _try_pytorch(self, pth_path: str, infer_config: InferenceConfig):
        """尝试创建 PyTorch 推理器"""
        logger.note(f"尝试使用 PyTorch: {Path(pth_path).name}")
        inferencer = PyTorchInferencer(pth_path, config=infer_config)
        logger.okay(f"PyTorch 推理器初始化成功")
        return inferencer, "PyTorch"

    def _find_latest_model_base(self) -> Path:
        """
        查找最新的模型文件基础路径（不含扩展名）。

        按优先级查找: .engine > .onnx > .pth
        """
        # 查找所有 _best 模型文件
        all_models = []
        for suffix in [".engine", ".onnx", ".pth"]:
            all_models.extend(CKPT_DIR.glob(f"*_best{suffix}"))

        if not all_models:
            raise FileNotFoundError(f"在 {CKPT_DIR} 中未找到任何模型文件")

        # 按修改时间排序，获取最新的
        all_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = all_models[0]

        # 返回不含扩展名的路径
        model_base = latest.with_suffix("")
        logger.note(f"找到模型: {latest.name}")
        return model_base

    def _capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧"""
        return self.frame_capturer.capture_tensor()

    def _infer(self) -> Optional[dict]:
        """执行推理"""
        if len(self.frame_buffer) < self.config.history_frames:
            return None

        # 准备输入数据
        # images: (batch, history_frames, 3, H, W)
        frames = list(self.frame_buffer)
        images = np.stack(frames, axis=0)  # (history_frames, 3, H, W)
        images = images[np.newaxis, ...]  # (1, history_frames, 3, H, W)

        # key_history: (batch, history_frames - 1, num_keys)
        key_hist = None
        if len(self.key_history) == self.config.history_frames - 1:
            key_hist = np.stack(list(self.key_history), axis=0)  # (history_frames-1, 4)
            key_hist = key_hist[np.newaxis, ...].astype(np.float32)

        # 推理
        start_time = time.perf_counter()
        result = self.inferencer.infer(images, key_hist)
        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)

        return result

    def _apply_control(self, preds: np.ndarray):
        """应用控制输出到手柄"""
        # 映射为摇杆方向
        direction = self.mapper.map_from_prediction(preds.flatten())

        # 应用到手柄
        self.gamepad.move_left_joystick(direction)

    def _update_key_history(self, preds: np.ndarray):
        """更新按键历史"""
        self.key_history.append(preds.flatten())

    def _format_prediction(self, preds: np.ndarray, probs: np.ndarray) -> str:
        """格式化预测结果用于显示"""
        keys = []
        for i in range(NUM_KEYS):
            if preds.flatten()[i] > 0.5:
                prob = probs.flatten()[i]
                keys.append(f"{INDEX_TO_KEY[i]}({prob:.2f})")

        if not keys:
            return "无"
        return " ".join(keys)

    def _control_loop(self):
        """控制循环"""
        logger.note(f"控制循环启动，帧率: {self.config.fps} FPS")

        next_tick = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # 捕获帧
            frame = self._capture_frame()
            if frame is None:
                logger.warn("帧捕获失败")
                time.sleep(0.1)
                continue

            # 添加到缓冲区
            self.frame_buffer.append(frame)
            self._frame_count += 1

            # 推理
            result = self._infer()
            if result is not None:
                preds = result["preds"]
                probs = result["probs"]

                # 应用控制
                self._apply_control(preds)

                # 更新按键历史
                self._update_key_history(preds)

                # 每隔一段时间打印统计信息
                if self._frame_count % (int(self.config.fps) * 2) == 0:
                    avg_infer_time = np.mean(list(self._inference_times))
                    pred_str = self._format_prediction(preds, probs)
                    logger.mesg(
                        f"帧: {self._frame_count}, "
                        f"推理: {avg_infer_time:.1f}ms, "
                        f"预测: {pred_str}"
                    )

            # 动态时间补偿
            next_tick += self.config.interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 如果落后太多，重置定时器
                if sleep_time < -self.config.interval:
                    next_tick = time.perf_counter()

        # 停止时回中摇杆
        self.gamepad.return_left_joystick_to_center()
        logger.note("控制循环结束")

    def start(self):
        """启动控制"""
        if self._running:
            logger.warn("控制器已在运行")
            return

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        logger.okay("控制器已启动")

    def stop(self):
        """停止控制"""
        if not self._running:
            return

        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

        # 确保摇杆回中
        self.gamepad.return_left_joystick_to_center()
        logger.okay("控制器已停止")

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running


# ===== 交互式控制器 ===== #


class InteractiveRealtimeController:
    """交互式实时控制器，支持按键启动/停止"""

    def __init__(self, config: RealtimeConfig = None):
        """
        初始化交互式控制器。

        :param config: 配置对象
        """
        self.controller = RealtimeController(config)
        self.user32 = ctypes.windll.user32

        # 按键状态跟踪（用于边沿检测）
        self._last_start_key_pressed = False
        self._last_stop_key_pressed = False

    def _is_key_pressed(self, vk_code: int) -> bool:
        """检查按键是否按下"""
        return bool(self.user32.GetAsyncKeyState(vk_code) & 0x8000)

    def _check_control_keys(self) -> Optional[str]:
        """
        检查控制键状态。

        :return: "start", "stop" 或 None
        """
        # 检测启动键的上升沿
        start_pressed = self._is_key_pressed(START_KEY_VK)
        if start_pressed and not self._last_start_key_pressed:
            self._last_start_key_pressed = start_pressed
            return "start"
        self._last_start_key_pressed = start_pressed

        # 检测停止键的上升沿
        stop_pressed = self._is_key_pressed(STOP_KEY_VK)
        if stop_pressed and not self._last_stop_key_pressed:
            self._last_stop_key_pressed = stop_pressed
            return "stop"
        self._last_stop_key_pressed = stop_pressed

        return None

    def run(self):
        """运行交互式控制循环"""
        logger.note("=" * 60)
        logger.note("GTAV 实时行为克隆控制")
        logger.note("=" * 60)
        logger.hint(f"按 '3' 键启动控制")
        logger.hint(f"按 '4' 键停止控制")
        logger.hint(f"按 Ctrl+C 退出程序")
        logger.note(f"推理后端: {self.controller.backend_name}")
        logger.note("=" * 60)

        try:
            while True:
                action = self._check_control_keys()

                if action == "start" and not self.controller.is_running():
                    logger.okay("检测到启动键，开始控制...")
                    self.controller.start()
                elif action == "stop" and self.controller.is_running():
                    logger.warn("检测到停止键，停止控制...")
                    self.controller.stop()

                time.sleep(0.05)  # 20Hz 检测频率

        except KeyboardInterrupt:
            logger.warn("\n检测到 Ctrl+C，正在退出...")
        finally:
            if self.controller.is_running():
                self.controller.stop()
            logger.okay("程序已退出")


# ===== 参数解析器 ===== #


class RealtimeArgParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="GTAV 实时行为克隆控制",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  # 使用默认模型运行（自动选择最佳后端）
  python -m gtaz.agency_move.realtime

  # 指定模型路径（支持 .engine, .onnx, .pth）
  python -m gtaz.agency_move.realtime -m path/to/model.engine
  python -m gtaz.agency_move.realtime -m path/to/model.onnx
  python -m gtaz.agency_move.realtime -m path/to/model.pth

  # 设置帧率和推理阈值
  python -m gtaz.agency_move.realtime -f 10 -t 0.5
""",
        )

        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=None,
            help="模型路径，支持 .engine (TensorRT), .onnx (ONNX Runtime), .pth (PyTorch)。默认自动查找最新模型并选择最佳后端",
        )

        parser.add_argument(
            "-f",
            "--fps",
            type=float,
            default=DEFAULT_FPS,
            help=f"帧率（默认 {DEFAULT_FPS}）",
        )

        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=DEFAULT_THRESHOLD,
            help=f"推理阈值（默认 {DEFAULT_THRESHOLD}）",
        )

        parser.add_argument(
            "-s",
            "--strength",
            type=float,
            default=1.0,
            help="摇杆推动强度 (0.0-1.0)，默认 1.0",
        )

        parser.add_argument(
            "--history-frames",
            type=int,
            default=DEFAULT_HISTORY_FRAMES,
            help=f"历史帧数（默认 {DEFAULT_HISTORY_FRAMES}）",
        )

        return parser

    def parse_args(self):
        return self.parser.parse_args()

    def create_config(self, args) -> RealtimeConfig:
        return RealtimeConfig(
            fps=args.fps,
            threshold=args.threshold,
            history_frames=args.history_frames,
            joystick_strength=args.strength,
            model_path=args.model,
        )


# ===== 主函数 ===== #


def main():
    """主函数"""
    arg_parser = RealtimeArgParser()
    args = arg_parser.parse_args()
    config = arg_parser.create_config(args)

    # 创建并运行交互式控制器
    controller = InteractiveRealtimeController(config)
    controller.run()


if __name__ == "__main__":
    with Runtimer():
        main()

    # 使用示例:

    # Case: 使用默认参数运行（自动选择最佳后端）
    # python -m gtaz.agency_move.realtime

    # Case: 指定帧率和阈值
    # python -m gtaz.agency_move.realtime -f 15 -t 0.4

    # Case: 指定 TensorRT 模型
    # python -m gtaz.agency_move.realtime -m path/to/model.engine

    # Case: 指定 ONNX 模型
    # python -m gtaz.agency_move.realtime -m path/to/model.onnx

    # Case: 指定 PyTorch 模型
    # python -m gtaz.agency_move.realtime -m path/to/model.pth
