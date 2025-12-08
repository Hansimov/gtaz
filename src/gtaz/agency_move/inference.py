"""
GTAV 行为克隆模型推理模块

支持 PyTorch (.pth)、ONNX Runtime 和 TensorRT (.engine) 三种推理方式，并进行性能对比。

## 主要功能:

1. 模型导出和转换: PyTorch .pth → ONNX → TensorRT .engine
2. 模型推理: 支持 PyTorch、ONNX Runtime 和 TensorRT 三种推理后端
3. 性能对比: 对比三种推理后端的速度

## 性能对比结果 (RTX 2060 SUPER, batch_size=1):
- PyTorch:      ~5.08 ms (196.8 FPS) - 1.00x
- ONNX Runtime: ~4.77 ms (209.8 FPS) - 1.07x
- TensorRT:     ~2.20 ms (455.3 FPS) - 2.31x

## 依赖安装:

```sh
# 安装PyTorch：Windows下安装CUDA版本需要指定--index-url
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装ONNX：卸载安装的CPU版本和旧的GPU版本，重新安装GPU版本
pip uninstall onnx onnxruntime onnxscript onnxruntime-gpu -y
pip install --upgrade onnx onnxscript onnxruntime-gpu

# 安装TensorRT：卸载冲突的cu13版本，重新安装 cu12 版本：
pip uninstall tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs -y
pip uninstall tensorrt-cu12 tensorrt_cu12_bindings tensorrt_cu12_libs -y
pip install --upgrade tensorrt-cu12
```

或者检查是否已正确安装：

```sh
pip list | findstr tensorrt
# 如果后续想查看更多：
# pip list | findstr -i "tensor cuda nvidia"
```

应当输出如下内容：（不包含cu13的信息）

```sh
tensorrt_cu12            10.14.1.48.post1
tensorrt_cu12_bindings   10.14.1.48.post1
tensorrt_cu12_libs       10.14.1.48.post1
torch_tensorrt           2.9.0
```

## 注意事项：
1. cuda-python 包在新版本中 API 有变化，本模块使用 PyTorch 的 CUDA 接口
进行 GPU 内存管理，避免了 cuda-python 的兼容性问题。
2. TensorRT engine 与 GPU 架构绑定，更换 GPU 后需要重新构建。
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tclogger import TCLogger, logstr, dict_to_str, Runtimer
from torchvision import transforms

# 尝试导入 TensorRT 相关依赖
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    print(
        f"Warning: TensorRT not available. TensorRT inference will be disabled. ({e})"
    )

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print(
        "Warning: ONNX or ONNXRuntime not available. ONNX export/validation will be disabled."
    )


logger = TCLogger(name="AgencyMoveInfer", use_prefix=True)


# ===== 常量定义 ===== #

KEY_TO_INDEX = {"W": 0, "A": 1, "S": 2, "D": 3}
INDEX_TO_KEY = {0: "W", 1: "A", 2: "S", 3: "D"}
NUM_KEYS = 4

# 目录
SRC_DIR = Path(__file__).parent.parent
CKPT_DIR = SRC_DIR / "checkpoints/agency_move"
DATA_DIR = SRC_DIR / "cache/agency_move"


# ===== 配置类 ===== #


@dataclass
class InferenceConfig:
    """推理配置"""

    # 模型参数
    history_frames: int = 4
    num_keys: int = NUM_KEYS
    hidden_dim: int = 256
    use_key_history: bool = True
    dropout: float = 0.3
    image_size: tuple[int, int] = (160, 220)  # (H, W)

    # 推理参数
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # TensorRT 参数
    fp16: bool = True
    max_workspace_size: int = 4 << 30  # 4GB

    # 动态 shape 参数 (batch_size, history_frames, channels, height, width)
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 8


# ===== 模型定义 (从 train.py 复制，用于加载) ===== #


class TemporalResNet(nn.Module):
    """时序 ResNet 模型：ResNet18 + GRU + 历史按键嵌入"""

    def __init__(
        self,
        history_frames: int = 4,
        num_keys: int = NUM_KEYS,
        hidden_dim: int = 256,
        use_key_history: bool = True,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision import models

        self.history_frames = history_frames
        self.num_keys = num_keys
        self.use_key_history = use_key_history
        self.hidden_dim = hidden_dim
        self.feature_dim = 512

        # ResNet18 backbone
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 时序融合层
        self.temporal_gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 历史按键嵌入
        fc_input_dim = hidden_dim
        if use_key_history:
            self.key_embedding = nn.Sequential(
                nn.Linear((history_frames - 1) * num_keys, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fc_input_dim += 64
        else:
            self.key_embedding = None

        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_keys),
        )

    def forward(
        self, images: torch.Tensor, key_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_frames_in_batch, C, H, W = images.shape

        # 提取特征
        images_flat = images.view(batch_size * num_frames_in_batch, C, H, W)
        features = self.backbone(images_flat)
        features = self.avgpool(features).view(batch_size * num_frames_in_batch, -1)
        features = features.view(batch_size, num_frames_in_batch, self.feature_dim)

        # GRU 时序建模
        _, hidden = self.temporal_gru(features)
        temporal_features = hidden.squeeze(0)

        # 融合按键历史
        if self.use_key_history and key_history is not None:
            key_flat = key_history.view(batch_size, -1)
            key_features = self.key_embedding(key_flat)
            combined = torch.cat([temporal_features, key_features], dim=1)
        else:
            combined = temporal_features

        return self.fc(combined)


# ===== 第一部分：模型导出和转换 ===== #


class ModelExporter:
    """模型导出器：PyTorch .pth → ONNX → TensorRT"""

    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)

    def load_pytorch_model(self, model_path: str) -> tuple[nn.Module, dict]:
        """加载 PyTorch 模型"""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        config_dict = checkpoint.get("config", {})

        model = TemporalResNet(
            history_frames=config_dict.get(
                "history_frames", self.config.history_frames
            ),
            num_keys=NUM_KEYS,
            hidden_dim=config_dict.get("hidden_dim", self.config.hidden_dim),
            use_key_history=config_dict.get(
                "use_key_history", self.config.use_key_history
            ),
            dropout=config_dict.get("dropout", self.config.dropout),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        # 更新配置
        self.config.history_frames = config_dict.get(
            "history_frames", self.config.history_frames
        )
        self.config.hidden_dim = config_dict.get("hidden_dim", self.config.hidden_dim)
        self.config.use_key_history = config_dict.get(
            "use_key_history", self.config.use_key_history
        )
        if "image_size" in config_dict:
            self.config.image_size = tuple(config_dict["image_size"])

        logger.okay(f"加载 PyTorch 模型: {model_path}")
        return model, config_dict

    def export_to_onnx(
        self, model: nn.Module, onnx_path: str, dynamic_batch: bool = True
    ) -> str:
        """导出模型到 ONNX 格式"""
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "ONNX not available. Please install: pip install onnx onnxscript"
            )

        model.eval()
        H, W = self.config.image_size
        batch_size = 1

        # 创建示例输入
        dummy_images = torch.randn(
            batch_size, self.config.history_frames, 3, H, W, device=self.device
        )

        # 根据是否使用 key_history 创建输入
        if self.config.use_key_history:
            # key_history shape: (batch, history_frames - 1, num_keys)
            # 因为 key_history 不包含当前帧的按键（是要预测的目标）
            dummy_key_history = torch.randn(
                batch_size, self.config.history_frames - 1, NUM_KEYS, device=self.device
            )
            dummy_inputs = (dummy_images, dummy_key_history)
            input_names = ["images", "key_history"]

            dynamic_axes = (
                {
                    "images": {0: "batch"},
                    "key_history": {0: "batch"},
                    "output": {0: "batch"},
                }
                if dynamic_batch
                else None
            )
        else:
            dummy_inputs = (dummy_images,)
            input_names = ["images"]
            dynamic_axes = (
                {
                    "images": {0: "batch"},
                    "output": {0: "batch"},
                }
                if dynamic_batch
                else None
            )

        # 导出 ONNX (使用旧版 API 以避免 GRU 与 torch.export 的兼容性问题)
        logger.note(f"导出 ONNX 模型到: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=17,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,  # 使用旧版 TorchScript 导出，避免 GRU 兼容性问题
        )

        # 验证 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.okay(f"ONNX 模型验证通过: {onnx_path}")

        return onnx_path

    def validate_onnx(self, onnx_path: str, model: nn.Module) -> bool:
        """验证 ONNX 模型输出与 PyTorch 模型一致"""
        if not ONNX_AVAILABLE:
            logger.warn("ONNX Runtime 不可用，跳过验证")
            return False

        H, W = self.config.image_size

        # 创建测试输入
        test_images = torch.randn(
            1, self.config.history_frames, 3, H, W, device=self.device
        )
        test_key_history = torch.randn(
            1, self.config.history_frames - 1, NUM_KEYS, device=self.device
        )

        # PyTorch 推理
        model.eval()
        with torch.no_grad():
            if self.config.use_key_history:
                pt_output = model(test_images, test_key_history)
            else:
                pt_output = model(test_images)
        pt_output = pt_output.cpu().numpy()

        # ONNX Runtime 推理
        session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        if self.config.use_key_history:
            ort_inputs = {
                "images": test_images.cpu().numpy(),
                "key_history": test_key_history.cpu().numpy(),
            }
        else:
            ort_inputs = {"images": test_images.cpu().numpy()}

        ort_output = session.run(None, ort_inputs)[0]

        # 对比输出
        diff = np.abs(pt_output - ort_output).max()
        is_close = np.allclose(pt_output, ort_output, rtol=1e-3, atol=1e-5)

        if is_close:
            logger.okay(f"ONNX 模型验证通过，最大误差: {diff:.6f}")
        else:
            logger.warn(f"ONNX 模型输出差异较大，最大误差: {diff:.6f}")

        return is_close

    def build_tensorrt_engine(
        self,
        onnx_path: str,
        engine_path: str,
    ) -> str:
        """从 ONNX 构建 TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT not available. Please install: pip install tensorrt-cu12 cuda-python"
            )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        H, W = self.config.image_size

        # 定义输入形状
        # images: (batch, history_frames, 3, H, W)
        images_min = (self.config.min_batch_size, self.config.history_frames, 3, H, W)
        images_opt = (self.config.opt_batch_size, self.config.history_frames, 3, H, W)
        images_max = (self.config.max_batch_size, self.config.history_frames, 3, H, W)

        # key_history: (batch, history_frames - 1, num_keys)
        key_min = (self.config.min_batch_size, self.config.history_frames - 1, NUM_KEYS)
        key_opt = (self.config.opt_batch_size, self.config.history_frames - 1, NUM_KEYS)
        key_max = (self.config.max_batch_size, self.config.history_frames - 1, NUM_KEYS)

        logger.note(f"构建 TensorRT engine: {engine_path}")
        logger.mesg(f"  - FP16: {self.config.fp16}")
        logger.mesg(
            f"  - Batch size: min={self.config.min_batch_size}, opt={self.config.opt_batch_size}, max={self.config.max_batch_size}"
        )

        # 创建 builder 和 network
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            explicit_batch_flag
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            # 解析 ONNX
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    logger.err("ONNX 解析失败:")
                    for i in range(parser.num_errors):
                        logger.err(f"  {parser.get_error(i)}")
                    raise RuntimeError("ONNX parse failed")

            logger.okay("ONNX 解析成功")

            # 创建配置
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.config.max_workspace_size
            )

            # 启用 FP16
            if self.config.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.note("启用 FP16 精度")

            # 创建优化 profile (用于动态 shape)
            profile = builder.create_optimization_profile()

            # 设置 images 输入的动态 shape
            images_input = network.get_input(0)
            profile.set_shape(images_input.name, images_min, images_opt, images_max)

            # 如果有 key_history 输入，设置其动态 shape
            if network.num_inputs > 1:
                key_input = network.get_input(1)
                profile.set_shape(key_input.name, key_min, key_opt, key_max)

            config.add_optimization_profile(profile)

            # 构建序列化 engine
            logger.note("开始构建 TensorRT engine (这可能需要几分钟)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # 保存 engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.okay(f"TensorRT engine 保存到: {engine_path}")
            return engine_path

    def convert_model(
        self, pth_path: str, output_dir: str = None, skip_if_exists: bool = True
    ) -> tuple[str, str]:
        """完整的模型转换流程: .pth → .onnx → .engine"""
        pth_path = Path(pth_path)
        output_dir = Path(output_dir) if output_dir else pth_path.parent

        base_name = pth_path.stem
        onnx_path = output_dir / f"{base_name}.onnx"
        engine_path = output_dir / f"{base_name}.engine"

        # 加载 PyTorch 模型
        model, config_dict = self.load_pytorch_model(str(pth_path))

        # 导出 ONNX
        if skip_if_exists and onnx_path.exists():
            logger.note(f"ONNX 文件已存在，跳过导出: {onnx_path}")
        else:
            self.export_to_onnx(model, str(onnx_path))
            self.validate_onnx(str(onnx_path), model)

        # 构建 TensorRT engine
        if skip_if_exists and engine_path.exists():
            logger.note(f"TensorRT engine 已存在，跳过构建: {engine_path}")
        else:
            if TENSORRT_AVAILABLE:
                self.build_tensorrt_engine(str(onnx_path), str(engine_path))
            else:
                logger.warn("TensorRT 不可用，跳过 engine 构建")

        return str(onnx_path), str(engine_path)


# ===== 第二部分：模型推理 ===== #


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, image_size: tuple[int, int] = (160, 220)):
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_path: str) -> torch.Tensor:
        """预处理单张图像"""
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def preprocess_batch(self, image_paths: list[str]) -> torch.Tensor:
        """预处理多张图像"""
        images = [self.preprocess(p) for p in image_paths]
        return torch.stack(images, dim=0)


class PyTorchInferencer:
    """PyTorch 推理器"""

    def __init__(self, model_path: str, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.model, self.model_config = self._load_model(model_path)
        self.preprocessor = ImagePreprocessor(self.config.image_size)

    def _load_model(self, model_path: str) -> tuple[nn.Module, dict]:
        """加载模型"""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        config_dict = checkpoint.get("config", {})

        # 更新配置
        self.config.history_frames = config_dict.get(
            "history_frames", self.config.history_frames
        )
        self.config.use_key_history = config_dict.get(
            "use_key_history", self.config.use_key_history
        )
        if "image_size" in config_dict:
            self.config.image_size = tuple(config_dict["image_size"])

        model = TemporalResNet(
            history_frames=self.config.history_frames,
            num_keys=NUM_KEYS,
            hidden_dim=config_dict.get("hidden_dim", self.config.hidden_dim),
            use_key_history=self.config.use_key_history,
            dropout=config_dict.get("dropout", self.config.dropout),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        return model, config_dict

    @torch.no_grad()
    def infer(
        self, images: torch.Tensor, key_history: Optional[torch.Tensor] = None
    ) -> dict:
        """推理"""
        images = images.to(self.device)
        if key_history is not None:
            key_history = key_history.to(self.device)

        if self.config.use_key_history and key_history is not None:
            logits = self.model(images, key_history)
        else:
            logits = self.model(images)

        probs = torch.sigmoid(logits)
        preds = (probs > self.config.threshold).float()

        return {
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "preds": preds.cpu().numpy(),
        }

    def infer_from_files(
        self, image_paths: list[str], key_history: Optional[np.ndarray] = None
    ) -> dict:
        """从文件推理

        Args:
            image_paths: 图像路径列表，长度为 history_frames
            key_history: 按键历史，shape (history_frames, num_keys)，会自动取 [:-1]
        """
        # 预处理图像
        images = self.preprocessor.preprocess_batch(image_paths)
        images = images.unsqueeze(0)  # 添加 batch 维度

        # 准备 key_history (取 [:-1]，因为最后一帧是要预测的目标)
        if key_history is not None:
            key_history = torch.from_numpy(key_history[:-1]).float().unsqueeze(0)

        return self.infer(images, key_history)


class TensorRTInferencer:
    """TensorRT 推理器"""

    def __init__(self, engine_path: str, config: InferenceConfig = None):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.config = config or InferenceConfig()
        self.engine_path = engine_path
        self.preprocessor = ImagePreprocessor(self.config.image_size)

        # 加载 engine
        self.engine, self.context = self._load_engine()

        # 获取输入输出信息
        self._setup_io()

    def _load_engine(self):
        """加载 TensorRT engine"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        return engine, context

    def _setup_io(self):
        """设置输入输出绑定"""
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.note(f"TensorRT engine 输入: {self.input_names}")
        logger.note(f"TensorRT engine 输出: {self.output_names}")

    def infer(
        self, images: np.ndarray, key_history: Optional[np.ndarray] = None
    ) -> dict:
        """推理

        Args:
            images: shape (batch, history_frames, 3, H, W), dtype float32
            key_history: shape (batch, history_frames - 1, num_keys), dtype float32
        """
        batch_size = images.shape[0]

        # 设置输入 shape
        H, W = self.config.image_size
        images_shape = (batch_size, self.config.history_frames, 3, H, W)
        self.context.set_input_shape("images", images_shape)

        if key_history is not None and len(self.input_names) > 1:
            key_shape = (batch_size, self.config.history_frames - 1, NUM_KEYS)
            self.context.set_input_shape("key_history", key_shape)

        # 使用 PyTorch 管理 GPU 内存（更稳定）
        device = torch.device("cuda")

        # 转换为 PyTorch tensor 并移到 GPU
        images_tensor = (
            torch.from_numpy(images.astype(np.float32)).contiguous().to(device)
        )

        if key_history is not None and len(self.input_names) > 1:
            key_tensor = (
                torch.from_numpy(key_history.astype(np.float32)).contiguous().to(device)
            )

        # 获取输出形状并分配内存
        output_shape = tuple(self.context.get_tensor_shape("output"))
        output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)

        # 设置 tensor 地址
        self.context.set_tensor_address("images", images_tensor.data_ptr())
        if key_history is not None and len(self.input_names) > 1:
            self.context.set_tensor_address("key_history", key_tensor.data_ptr())
        self.context.set_tensor_address("output", output_tensor.data_ptr())

        # 执行推理
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # 获取输出
        output = output_tensor.cpu().numpy()

        # 后处理
        probs = 1 / (1 + np.exp(-output))  # sigmoid
        preds = (probs > self.config.threshold).astype(np.float32)

        return {
            "logits": output,
            "probs": probs,
            "preds": preds,
        }

    def infer_from_files(
        self, image_paths: list[str], key_history: Optional[np.ndarray] = None
    ) -> dict:
        """从文件推理

        Args:
            image_paths: 图像路径列表，长度为 history_frames
            key_history: 按键历史，shape (history_frames, num_keys)，会自动取 [:-1]
        """
        # 预处理图像
        images = self.preprocessor.preprocess_batch(image_paths)
        images = images.unsqueeze(0).numpy()  # 添加 batch 维度

        # 准备 key_history (取 [:-1]，因为最后一帧是要预测的目标)
        if key_history is not None:
            key_history = key_history[:-1][
                np.newaxis, ...
            ]  # (1, history_frames-1, num_keys)

        return self.infer(images, key_history)


class ONNXRuntimeInferencer:
    """ONNX Runtime 推理器（作为 TensorRT 的备选高效推理方案）"""

    def __init__(self, onnx_path: str, config: InferenceConfig = None):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        self.config = config or InferenceConfig()
        self.onnx_path = onnx_path
        self.preprocessor = ImagePreprocessor(self.config.image_size)

        # 选择最佳的 execution provider
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.okay("ONNX Runtime 使用 CUDA 执行")
        else:
            self.providers = ["CPUExecutionProvider"]
            logger.warn("ONNX Runtime 使用 CPU 执行（无 CUDA 支持）")

        self.session = ort.InferenceSession(onnx_path, providers=self.providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(
        self, images: np.ndarray, key_history: Optional[np.ndarray] = None
    ) -> dict:
        """推理

        Args:
            images: shape (batch, history_frames, 3, H, W), dtype float32
            key_history: shape (batch, history_frames - 1, num_keys), dtype float32
        """
        # 准备输入
        images = np.ascontiguousarray(images.astype(np.float32))

        if (
            self.config.use_key_history
            and key_history is not None
            and len(self.input_names) > 1
        ):
            key_history = np.ascontiguousarray(key_history.astype(np.float32))
            ort_inputs = {
                "images": images,
                "key_history": key_history,
            }
        else:
            ort_inputs = {"images": images}

        # 推理
        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]

        # 后处理
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs > self.config.threshold).astype(np.float32)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }

    def infer_from_files(
        self, image_paths: list[str], key_history: Optional[np.ndarray] = None
    ) -> dict:
        """从文件推理

        Args:
            image_paths: 图像路径列表，长度为 history_frames
            key_history: 按键历史，shape (history_frames, num_keys)，会自动取 [:-1]
        """
        # 预处理图像
        images = self.preprocessor.preprocess_batch(image_paths)
        images = images.unsqueeze(0).numpy()  # 添加 batch 维度

        # 准备 key_history (取 [:-1]，因为最后一帧是要预测的目标)
        if key_history is not None:
            key_history = key_history[:-1][
                np.newaxis, ...
            ]  # (1, history_frames-1, num_keys)

        return self.infer(images, key_history)


# ===== 第三部分：性能对比 ===== #


class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(
        self,
        pth_path: str,
        onnx_path: str = None,
        engine_path: str = None,
        config: InferenceConfig = None,
    ):
        self.config = config or InferenceConfig()
        self.pth_path = pth_path
        self.onnx_path = onnx_path
        self.engine_path = engine_path

        # 初始化推理器
        self.pt_inferencer = PyTorchInferencer(pth_path, self.config)

        # ONNX Runtime 推理器
        if onnx_path and Path(onnx_path).exists() and ONNX_AVAILABLE:
            self.ort_inferencer = ONNXRuntimeInferencer(onnx_path, self.config)
        else:
            self.ort_inferencer = None
            if not ONNX_AVAILABLE:
                logger.warn("ONNX Runtime 不可用，跳过 ONNX 性能测试")

        # TensorRT 推理器
        if engine_path and Path(engine_path).exists() and TENSORRT_AVAILABLE:
            self.trt_inferencer = TensorRTInferencer(engine_path, self.config)
        else:
            self.trt_inferencer = None
            if not TENSORRT_AVAILABLE:
                logger.warn("TensorRT 不可用，跳过 TensorRT 性能测试")

    def _create_dummy_input(
        self, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """创建测试输入"""
        H, W = self.config.image_size
        images = torch.randn(
            batch_size, self.config.history_frames, 3, H, W, dtype=torch.float32
        )
        key_history = torch.randn(
            batch_size, self.config.history_frames - 1, NUM_KEYS, dtype=torch.float32
        )
        return images, key_history

    def benchmark_pytorch(
        self, num_iterations: int = 100, warmup: int = 10, batch_size: int = 1
    ) -> dict:
        """PyTorch 性能测试"""
        images, key_history = self._create_dummy_input(batch_size)
        images = images.to(self.pt_inferencer.device)
        key_history = key_history.to(self.pt_inferencer.device)

        # 预热
        for _ in range(warmup):
            self.pt_inferencer.infer(images, key_history)

        # 同步 CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 计时
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.pt_inferencer.infer(images, key_history)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000.0 / np.mean(times) * batch_size,
        }

    def benchmark_onnxruntime(
        self, num_iterations: int = 100, warmup: int = 10, batch_size: int = 1
    ) -> Optional[dict]:
        """ONNX Runtime 性能测试"""
        if self.ort_inferencer is None:
            return None

        images, key_history = self._create_dummy_input(batch_size)
        images_np = images.numpy()
        key_history_np = key_history.numpy()

        # 预热
        for _ in range(warmup):
            self.ort_inferencer.infer(images_np, key_history_np)

        # 计时
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.ort_inferencer.infer(images_np, key_history_np)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000.0 / np.mean(times) * batch_size,
        }

    def benchmark_tensorrt(
        self, num_iterations: int = 100, warmup: int = 10, batch_size: int = 1
    ) -> Optional[dict]:
        """TensorRT 性能测试"""
        if self.trt_inferencer is None:
            return None

        images, key_history = self._create_dummy_input(batch_size)
        images_np = images.numpy()
        key_history_np = key_history.numpy()

        # 预热
        for _ in range(warmup):
            self.trt_inferencer.infer(images_np, key_history_np)

        # 计时
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.trt_inferencer.infer(images_np, key_history_np)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000.0 / np.mean(times) * batch_size,
        }

    def compare(
        self, num_iterations: int = 100, warmup: int = 10, batch_size: int = 1
    ) -> dict:
        """对比 PyTorch、ONNX Runtime 和 TensorRT 性能"""
        logger.note(f"\n{'='*60}")
        logger.note(f"性能对比测试")
        logger.note(f"{'='*60}")
        logger.mesg(f"迭代次数: {num_iterations}")
        logger.mesg(f"预热次数: {warmup}")
        logger.mesg(f"批次大小: {batch_size}")
        logger.note(f"{'='*60}\n")

        results = {
            "config": {
                "num_iterations": num_iterations,
                "warmup": warmup,
                "batch_size": batch_size,
            }
        }

        # PyTorch 测试
        logger.note("PyTorch 推理性能测试...")
        pt_results = self.benchmark_pytorch(num_iterations, warmup, batch_size)
        logger.okay("PyTorch 结果:")
        logger.mesg(
            f"  平均延迟: {pt_results['mean_ms']:.2f} ± {pt_results['std_ms']:.2f} ms"
        )
        logger.mesg(
            f"  最小/最大: {pt_results['min_ms']:.2f} / {pt_results['max_ms']:.2f} ms"
        )
        logger.mesg(f"  吞吐量: {pt_results['fps']:.1f} FPS")
        results["pytorch"] = pt_results

        # ONNX Runtime 测试
        ort_results = None
        ort_speedup = None

        if self.ort_inferencer is not None:
            logger.note("\nONNX Runtime 推理性能测试...")
            ort_results = self.benchmark_onnxruntime(num_iterations, warmup, batch_size)
            logger.okay("ONNX Runtime 结果:")
            logger.mesg(
                f"  平均延迟: {ort_results['mean_ms']:.2f} ± {ort_results['std_ms']:.2f} ms"
            )
            logger.mesg(
                f"  最小/最大: {ort_results['min_ms']:.2f} / {ort_results['max_ms']:.2f} ms"
            )
            logger.mesg(f"  吞吐量: {ort_results['fps']:.1f} FPS")

            # 计算加速比
            ort_speedup = pt_results["mean_ms"] / ort_results["mean_ms"]
            logger.okay(f"  相对 PyTorch 加速比: {ort_speedup:.2f}x")

        results["onnxruntime"] = ort_results
        results["ort_speedup"] = ort_speedup

        # TensorRT 测试
        trt_results = None
        trt_speedup = None

        if self.trt_inferencer is not None:
            logger.note("\nTensorRT 推理性能测试...")
            trt_results = self.benchmark_tensorrt(num_iterations, warmup, batch_size)
            logger.okay("TensorRT 结果:")
            logger.mesg(
                f"  平均延迟: {trt_results['mean_ms']:.2f} ± {trt_results['std_ms']:.2f} ms"
            )
            logger.mesg(
                f"  最小/最大: {trt_results['min_ms']:.2f} / {trt_results['max_ms']:.2f} ms"
            )
            logger.mesg(f"  吞吐量: {trt_results['fps']:.1f} FPS")

            # 计算加速比
            trt_speedup = pt_results["mean_ms"] / trt_results["mean_ms"]
            logger.okay(f"  相对 PyTorch 加速比: {trt_speedup:.2f}x")

        results["tensorrt"] = trt_results
        results["trt_speedup"] = trt_speedup

        # 打印总结
        logger.note(f"\n{'='*60}")
        logger.note("性能总结")
        logger.note(f"{'='*60}")
        logger.mesg(
            f"PyTorch:      {pt_results['mean_ms']:.2f} ms ({pt_results['fps']:.1f} FPS)"
        )
        if ort_results:
            logger.mesg(
                f"ONNX Runtime: {ort_results['mean_ms']:.2f} ms ({ort_results['fps']:.1f} FPS) - {ort_speedup:.2f}x"
            )
        if trt_results:
            logger.mesg(
                f"TensorRT:     {trt_results['mean_ms']:.2f} ms ({trt_results['fps']:.1f} FPS) - {trt_speedup:.2f}x"
            )
        logger.note(f"{'='*60}")

        return results

    def validate_outputs(self) -> bool:
        """验证 PyTorch、ONNX Runtime 和 TensorRT 输出一致性"""
        images, key_history = self._create_dummy_input(1)

        # PyTorch 推理
        pt_result = self.pt_inferencer.infer(images, key_history)
        all_passed = True

        # ONNX Runtime 验证
        if self.ort_inferencer is not None:
            ort_result = self.ort_inferencer.infer(images.numpy(), key_history.numpy())
            diff = np.abs(pt_result["logits"] - ort_result["logits"]).max()
            is_close = np.allclose(
                pt_result["logits"], ort_result["logits"], rtol=1e-3, atol=1e-5
            )

            if is_close:
                logger.okay(f"ONNX Runtime 输出验证通过，最大误差: {diff:.6f}")
            else:
                logger.warn(f"ONNX Runtime 输出差异较大，最大误差: {diff:.6f}")
                all_passed = False
        else:
            logger.warn("ONNX Runtime 推理器不可用，跳过 ONNX 输出验证")

        # TensorRT 验证
        if self.trt_inferencer is not None:
            trt_result = self.trt_inferencer.infer(images.numpy(), key_history.numpy())
            diff = np.abs(pt_result["logits"] - trt_result["logits"]).max()
            is_close = np.allclose(
                pt_result["logits"], trt_result["logits"], rtol=1e-2, atol=1e-3
            )

            if is_close:
                logger.okay(f"TensorRT 输出验证通过，最大误差: {diff:.6f}")
            else:
                logger.warn(f"TensorRT 输出差异较大，最大误差: {diff:.6f}")
                all_passed = False
        else:
            logger.warn("TensorRT 推理器不可用，跳过 TensorRT 输出验证")

        return all_passed


# ===== 工具函数 ===== #


def load_test_sequence(
    data_dir: str, num_frames: int = 4
) -> tuple[list[str], np.ndarray]:
    """加载测试序列"""
    data_dir = Path(data_dir)

    # 获取所有 session 目录
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        raise ValueError(f"No session directories found in {data_dir}")

    # 使用第一个 session
    session_dir = session_dirs[0]

    # 获取所有 jpg 文件并排序
    jpg_files = sorted(session_dir.glob("*.jpg"))
    if len(jpg_files) < num_frames:
        raise ValueError(f"Not enough frames in {session_dir}")

    # 选择连续的帧
    image_paths = [str(f) for f in jpg_files[:num_frames]]

    # 加载对应的 key_history
    key_history = np.zeros((num_frames, NUM_KEYS), dtype=np.float32)
    for i, img_path in enumerate(image_paths):
        json_path = Path(img_path).with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            if data.get("has_action", False) and "keys" in data:
                for key_info in data["keys"]:
                    key_name = key_info.get("key_name", "")
                    is_pressed = key_info.get("is_pressed", False)
                    if key_name in KEY_TO_INDEX and is_pressed:
                        key_history[i, KEY_TO_INDEX[key_name]] = 1.0

    return image_paths, key_history


def decode_prediction(preds: np.ndarray) -> list[str]:
    """解码预测结果"""
    keys = []
    for i, p in enumerate(preds.flatten()):
        if p > 0.5:
            keys.append(INDEX_TO_KEY[i])
    return keys if keys else ["None"]


# ===== 主函数 ===== #


def main():
    """主函数：演示完整的模型转换和推理流程"""

    # 配置
    config = InferenceConfig()

    # 模型路径
    pth_path = CKPT_DIR / "agency_move_temporal_f4_b64_e30_lr_auto_best.pth"

    if not pth_path.exists():
        logger.err(f"模型文件不存在: {pth_path}")
        return

    logger.note(f"\n{'='*60}")
    logger.note("GTAV 行为克隆模型推理演示")
    logger.note(f"{'='*60}\n")

    # ===== 第一部分：模型转换 ===== #
    logger.note("=" * 40)
    logger.note("第一部分：模型导出和转换")
    logger.note("=" * 40)

    exporter = ModelExporter(config)
    onnx_path, engine_path = exporter.convert_model(str(pth_path), skip_if_exists=True)

    # ===== 第二部分：推理演示 ===== #
    logger.note("\n" + "=" * 40)
    logger.note("第二部分：推理演示")
    logger.note("=" * 40)

    # 加载测试数据
    try:
        image_paths, key_history = load_test_sequence(
            str(DATA_DIR), config.history_frames
        )
        logger.okay(f"加载测试序列: {len(image_paths)} 帧")

        # PyTorch 推理
        pt_inferencer = PyTorchInferencer(str(pth_path), config)
        pt_result = pt_inferencer.infer_from_files(image_paths, key_history)

        logger.note("PyTorch 推理结果:")
        logger.mesg(f"  概率: {pt_result['probs']}")
        logger.mesg(f"  预测: {decode_prediction(pt_result['preds'])}")

        # ONNX Runtime 推理
        if ONNX_AVAILABLE and onnx_path and Path(onnx_path).exists():
            ort_inferencer = ONNXRuntimeInferencer(onnx_path, config)
            ort_result = ort_inferencer.infer_from_files(image_paths, key_history)

            logger.note("ONNX Runtime 推理结果:")
            logger.mesg(f"  概率: {ort_result['probs']}")
            logger.mesg(f"  预测: {decode_prediction(ort_result['preds'])}")

        # TensorRT 推理
        if TENSORRT_AVAILABLE and engine_path and Path(engine_path).exists():
            trt_inferencer = TensorRTInferencer(engine_path, config)
            trt_result = trt_inferencer.infer_from_files(image_paths, key_history)

            logger.note("TensorRT 推理结果:")
            logger.mesg(f"  概率: {trt_result['probs']}")
            logger.mesg(f"  预测: {decode_prediction(trt_result['preds'])}")
    except Exception as e:
        logger.warn(f"推理演示跳过: {e}")

    # ===== 第三部分：性能对比 ===== #
    logger.note("\n" + "=" * 40)
    logger.note("第三部分：性能对比")
    logger.note("=" * 40)

    benchmark = PerformanceBenchmark(
        str(pth_path),
        onnx_path=onnx_path,
        engine_path=engine_path if TENSORRT_AVAILABLE else None,
        config=config,
    )

    # 验证输出一致性
    benchmark.validate_outputs()

    # 性能对比
    results = benchmark.compare(num_iterations=100, warmup=10, batch_size=1)

    # 保存结果
    results_path = CKPT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        # 转换为可序列化格式
        serializable_results = {
            "pytorch": results.get("pytorch"),
            "onnxruntime": results.get("onnxruntime"),
            "tensorrt": results.get("tensorrt"),
            "ort_speedup": results.get("ort_speedup"),
            "trt_speedup": results.get("trt_speedup"),
            "config": results.get("config"),
        }
        json.dump(serializable_results, f, indent=2)
    logger.okay(f"性能测试结果保存到: {results_path}")


if __name__ == "__main__":
    with Runtimer():
        main()
