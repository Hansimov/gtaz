"""
GTAV 行为克隆模型推理模块

支持 PyTorch (.pth)、ONNX Runtime 和 TensorRT (.engine) 三种推理方式，并进行性能对比。

## 主要功能

1. 模型导出和转换: PyTorch .pth → ONNX → TensorRT .engine
2. 模型推理: 支持 PyTorch、ONNX Runtime 和 TensorRT 三种推理后端
3. 性能对比: 对比三种推理后端的速度

## 性能对比结果 (RTX 2060 SUPER, batch_size=1):

- PyTorch:      ~5.08 ms (196.8 FPS) - 1.00x
- ONNX Runtime: ~4.77 ms (209.8 FPS) - 1.07x
- TensorRT:     ~2.20 ms (455.3 FPS) - 2.31x

## 依赖安装

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

## 注意事项

1. 安装依赖时出现的大部分问题，都是版本冲突造成的，卸载冲突版本及其依赖，一般都能解决
2. TensorRT engine 与 GPU 架构绑定，更换 GPU 后需要重新构建
"""

import argparse
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

        logger.okay(f"加载 PyTorch 模型:")
        logger.file(f"{model_path}")
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
        logger.note(f"导出 ONNX 模型:")
        logger.file(f"{onnx_path}")
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
        logger.okay(f"ONNX 模型验证通过:")
        logger.file(f"{onnx_path}")

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

        logger.note(f"构建 TensorRT engine:")
        logger.file(f"{engine_path}")
        logger.mesg(f"- FP16: {self.config.fp16}")
        logger.mesg(
            f"- Batch size: min={self.config.min_batch_size}, opt={self.config.opt_batch_size}, max={self.config.max_batch_size}"
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

            logger.okay(f"保存 TensorRT engine:")
            logger.file(f"{engine_path}")
            return engine_path

    def convert_model(
        self, pth_path: str, output_dir: str = None, overwrite: bool = True
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
        if not overwrite and onnx_path.exists():
            logger.note(f"ONNX 文件已存在，跳过导出: {onnx_path}")
        else:
            self.export_to_onnx(model, str(onnx_path))
            self.validate_onnx(str(onnx_path), model)

        # 构建 TensorRT engine
        if not overwrite and engine_path.exists():
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
        else:
            # 如果没有 key_history 但模型需要它，设置为零张量
            if len(self.input_names) > 1:
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
        elif len(self.input_names) > 1:
            # 创建零张量作为占位符
            key_tensor = torch.zeros(
                (batch_size, self.config.history_frames - 1, NUM_KEYS),
                dtype=torch.float32,
                device=device,
            )

        # 获取输出形状并分配内存（必须在所有输入 shape 设置后）
        output_shape = tuple(self.context.get_tensor_shape("output"))
        output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)

        # 设置 tensor 地址
        self.context.set_tensor_address("images", images_tensor.data_ptr())
        if len(self.input_names) > 1:
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


# ===== 参数解析器 ===== #


class InferenceArgParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        """创建参数解析器"""
        parser = argparse.ArgumentParser(
            description="GTAV 行为克隆模型推理工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # ===== 基础参数 ===== #
        parser.add_argument(
            "-m",
            "--mode",
            type=lambda s: s.split(","),
            default="benchmark",
            help="运行模式。支持指定多个，用逗号分隔，优先级: export > infer > validate > benchmark，示例: --mode export,infer,validate,benchmark",
        )
        parser.add_argument(
            "-p",
            "--pth-path",
            type=str,
            default=None,
            help="PyTorch 模型路径 (.pth)。默认不指定，使用最新的 best 模型",
        )

        # ===== 模型参数 ===== #
        model_group = parser.add_argument_group("模型参数")
        model_group.add_argument(
            "--history-frames",
            type=int,
            default=4,
            help="历史帧数量 (默认: 4)",
        )
        model_group.add_argument(
            "--hidden-dim",
            type=int,
            default=256,
            help="隐藏层维度 (默认: 256)",
        )
        model_group.add_argument(
            "--no-key-history",
            action="store_true",
            help="不使用按键历史 (默认: False，即使用按键历史)",
        )
        model_group.add_argument(
            "--dropout",
            type=float,
            default=0.3,
            help="Dropout 概率 (默认: 0.3)",
        )
        model_group.add_argument(
            "--image-size",
            type=int,
            nargs=2,
            default=[160, 220],
            metavar=("H", "W"),
            help="图像尺寸 (高度 宽度) (默认: 160 220)",
        )

        # ===== 推理参数 ===== #
        infer_group = parser.add_argument_group("推理参数")
        infer_group.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="预测阈值 (默认: 0.5)",
        )
        infer_group.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            choices=["cuda", "cpu"],
            help="设备 (默认: cuda 如果可用，否则 cpu)",
        )
        infer_group.add_argument(
            "--data-dir",
            type=str,
            default=str(DATA_DIR),
            help=f"测试数据目录 (默认: {DATA_DIR})",
        )
        infer_group.add_argument(
            "--backend",
            type=lambda s: s.split(","),
            default="tensorrt",
            help="推理引擎 (默认: tensorrt）。可指定多个，用逗号分隔: torch (PyTorch), onnx (ONNX Runtime), tensorrt (TensorRT)。示例: --backend torch,onnx,tensorrt",
        )

        # ===== TensorRT 参数 ===== #
        trt_group = parser.add_argument_group("TensorRT 参数")
        trt_group.add_argument(
            "--no-fp16",
            action="store_true",
            help="不使用 FP16 精度 (默认: False，即使用 FP16)",
        )
        trt_group.add_argument(
            "--max-workspace-size",
            type=int,
            default=4,
            help="最大工作空间大小 (GB) (默认: 4)",
        )
        trt_group.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help="最小批次大小 (默认: 1)",
        )
        trt_group.add_argument(
            "--opt-batch-size",
            type=int,
            default=1,
            help="优化批次大小 (默认: 1)",
        )
        trt_group.add_argument(
            "--max-batch-size",
            type=int,
            default=8,
            help="最大批次大小 (默认: 8)",
        )

        # ===== 性能测试参数 ===== #
        bench_group = parser.add_argument_group("性能测试参数")
        bench_group.add_argument(
            "--iterations",
            type=int,
            default=100,
            help="性能测试迭代次数 (默认: 100)",
        )
        bench_group.add_argument(
            "--warmup",
            type=int,
            default=10,
            help="性能测试预热次数 (默认: 10)",
        )
        bench_group.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="性能测试批次大小 (默认: 1)",
        )

        # ===== 导出参数 ===== #
        export_group = parser.add_argument_group("导出参数")
        export_group.add_argument(
            "-w",
            "--overwrite",
            action="store_true",
            help="覆盖导出的 ONNX/Engine 文件(默认: False)",
        )
        export_group.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="输出目录 (默认: 与模型相同目录)",
        )

        return parser

    def parse_args(self):
        """解析命令行参数"""
        return self.parser.parse_args()

    def create_config(self, args) -> InferenceConfig:
        """根据参数创建配置"""
        config = InferenceConfig(
            history_frames=args.history_frames,
            num_keys=NUM_KEYS,
            hidden_dim=args.hidden_dim,
            use_key_history=not args.no_key_history,
            dropout=args.dropout,
            image_size=tuple(args.image_size),
            threshold=args.threshold,
            device=args.device,
            fp16=not args.no_fp16,
            max_workspace_size=args.max_workspace_size << 30,  # GB to bytes
            min_batch_size=args.min_batch_size,
            opt_batch_size=args.opt_batch_size,
            max_batch_size=args.max_batch_size,
        )
        return config


# ===== 主函数 ===== #


def run_export(args, config: InferenceConfig):
    """运行模型导出"""
    logger.note("=" * 60)
    logger.note("模式: 模型导出")
    logger.note("=" * 60)

    pth_path = args.pth_path
    if not Path(pth_path).exists():
        logger.err(f"模型文件不存在: {pth_path}")
        return

    exporter = ModelExporter(config)
    output_dir = args.output_dir if args.output_dir else None
    onnx_path, engine_path = exporter.convert_model(
        pth_path, output_dir=output_dir, overwrite=args.overwrite
    )
    logger.okay(f"导出完成:")
    logger.file(f"* {onnx_path}")
    logger.file(f"* {engine_path}")


def run_infer(args, config: InferenceConfig):
    """运行推理演示"""
    logger.note("=" * 60)
    logger.note("模式: 推理演示")
    logger.note("=" * 60)

    pth_path = args.pth_path
    onnx_path = str(Path(pth_path).with_suffix(".onnx"))
    engine_path = str(Path(pth_path).with_suffix(".engine"))

    if not Path(pth_path).exists():
        logger.err(f"模型文件不存在: {pth_path}")
        return

    # 获取要使用的推理引擎
    backends = args.backend if isinstance(args.backend, list) else [args.backend]

    # 验证所有 backend 是否有效
    valid_backends = {"torch", "onnx", "tensorrt"}
    invalid_backends = [b for b in backends if b not in valid_backends]
    if invalid_backends:
        logger.err(f"无效的推理引擎: {', '.join(invalid_backends)}")
        logger.mesg(f"有效的推理引擎: {', '.join(valid_backends)}")
        return

    # 加载测试数据
    try:
        image_paths, key_history = load_test_sequence(
            args.data_dir, config.history_frames
        )
        logger.okay(f"加载测试序列: {len(image_paths)} 帧")
        logger.mesg(f"推理引擎: {', '.join(backends)}\n")

        # PyTorch 推理
        if "torch" in backends:
            logger.note("PyTorch 推理:")
            pt_inferencer = PyTorchInferencer(pth_path, config)
            pt_result = pt_inferencer.infer_from_files(image_paths, key_history)
            logger.mesg(f"  概率: {pt_result['probs']}")
            logger.mesg(f"  预测: {decode_prediction(pt_result['preds'])}")

        # ONNX Runtime 推理
        if "onnx" in backends:
            if ONNX_AVAILABLE and Path(onnx_path).exists():
                logger.note("\nONNX Runtime 推理:")
                ort_inferencer = ONNXRuntimeInferencer(onnx_path, config)
                ort_result = ort_inferencer.infer_from_files(image_paths, key_history)
                logger.mesg(f"  概率: {ort_result['probs']}")
                logger.mesg(f"  预测: {decode_prediction(ort_result['preds'])}")
            else:
                logger.warn("ONNX Runtime 不可用或 ONNX 模型不存在，跳过 ONNX 推理")

        # TensorRT 推理
        if "tensorrt" in backends:
            if TENSORRT_AVAILABLE and Path(engine_path).exists():
                logger.note("\nTensorRT 推理:")
                trt_inferencer = TensorRTInferencer(engine_path, config)
                trt_result = trt_inferencer.infer_from_files(image_paths, key_history)
                logger.mesg(f"  概率: {trt_result['probs']}")
                logger.mesg(f"  预测: {decode_prediction(trt_result['preds'])}")
            else:
                logger.warn("TensorRT 不可用或 Engine 不存在，跳过 TensorRT 推理")

    except Exception as e:
        logger.warn(f"推理演示失败: {e}")
        import traceback

        traceback.print_exc()


def run_benchmark(args, config: InferenceConfig):
    """运行性能测试"""
    logger.note("=" * 60)
    logger.note("模式: 性能测试")
    logger.note("=" * 60)

    pth_path = args.pth_path
    onnx_path = str(Path(pth_path).with_suffix(".onnx"))
    engine_path = str(Path(pth_path).with_suffix(".engine"))

    if not Path(pth_path).exists():
        logger.err(f"模型文件不存在: {pth_path}")
        return

    benchmark = PerformanceBenchmark(
        pth_path,
        onnx_path=onnx_path if Path(onnx_path).exists() else None,
        engine_path=engine_path if Path(engine_path).exists() else None,
        config=config,
    )

    # 性能对比
    results = benchmark.compare(
        num_iterations=args.iterations,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )

    # 保存结果
    results_path = CKPT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        serializable_results = {
            "pytorch": results.get("pytorch"),
            "onnxruntime": results.get("onnxruntime"),
            "tensorrt": results.get("tensorrt"),
            "ort_speedup": results.get("ort_speedup"),
            "trt_speedup": results.get("trt_speedup"),
            "config": results.get("config"),
        }
        json.dump(serializable_results, f, indent=2)
    logger.okay(f"性能测试结果:")
    logger.file(f"{results_path}")


def run_validate(args, config: InferenceConfig):
    """运行输出验证"""
    logger.note("=" * 60)
    logger.note("模式: 输出验证")
    logger.note("=" * 60)

    pth_path = args.pth_path
    onnx_path = str(Path(pth_path).with_suffix(".onnx"))
    engine_path = str(Path(pth_path).with_suffix(".engine"))

    if not Path(pth_path).exists():
        logger.err(f"模型文件不存在: {pth_path}")
        return

    benchmark = PerformanceBenchmark(
        pth_path,
        onnx_path=onnx_path if Path(onnx_path).exists() else None,
        engine_path=engine_path if Path(engine_path).exists() else None,
        config=config,
    )

    # 验证输出一致性
    all_passed = benchmark.validate_outputs()

    if all_passed:
        logger.okay("所有模型输出验证通过！")
    else:
        logger.warn("部分模型输出验证失败，请检查日志")


def run_all(args, config: InferenceConfig):
    """运行所有功能"""
    logger.note("=" * 60)
    logger.note("模式: 完整演示 (导出 + 推理 + 验证 + 性能测试)")
    logger.note("=" * 60)
    print()

    # 1. 导出
    logger.note("[1/4] 模型导出")
    run_export(args, config)
    print()

    # 2. 推理
    logger.note("[2/4] 推理演示")
    run_infer(args, config)
    print()

    # 3. 验证
    logger.note("[3/4] 输出验证")
    run_validate(args, config)
    print()

    # 4. 性能测试
    logger.note("[4/4] 性能测试")
    run_benchmark(args, config)


def find_latest_best_model(ckpt_dir: Path) -> str:
    """在检查点目录中查找最新的 best 模型

    Args:
        ckpt_dir: 检查点目录

    Returns:
        最新的 best 模型路径

    Raises:
        FileNotFoundError: 如果没有找到 best 模型
    """
    # 查找所有 _best.pth 文件
    best_models = list(ckpt_dir.glob("*_best.pth"))

    if not best_models:
        raise FileNotFoundError(f"未在 {ckpt_dir} 中找到 _best.pth 模型文件")

    # 按修改时间排序，获取最新的
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_model = best_models[0]

    if len(best_models) > 1:
        logger.note(f"找到 {len(best_models)} 个 best 模型，使用最新的:")

    return str(latest_model)


def main():
    """主函数"""
    # 解析参数
    arg_parser = InferenceArgParser()
    args = arg_parser.parse_args()
    config = arg_parser.create_config(args)

    # 确定模型路径
    if args.pth_path is None:
        try:
            args.pth_path = find_latest_best_model(CKPT_DIR)
            logger.note(f"使用最新的 best 模型:")
            logger.file(f"{args.pth_path}")
        except FileNotFoundError as e:
            logger.err(str(e))
            return

    # mode 现在是列表，支持多个模式
    modes = args.mode if isinstance(args.mode, list) else [args.mode]

    # 验证所有模式是否有效
    valid_modes = {"export", "infer", "benchmark", "validate", "all"}
    invalid_modes = [m for m in modes if m not in valid_modes]
    if invalid_modes:
        logger.err(f"无效的模式: {', '.join(invalid_modes)}")
        logger.mesg(f"有效的模式: {', '.join(valid_modes)}")
        return

    # 如果包含 "all"，优先执行 all 模式
    if "all" in modes:
        run_all(args, config)
        return

    # 定义模式到函数的映射
    mode_handlers = {
        "export": run_export,
        "infer": run_infer,
        "validate": run_validate,
        "benchmark": run_benchmark,
    }

    # 定义执行顺序（固定顺序）
    mode_order = ["export", "infer", "validate", "benchmark"]

    # 过滤出用户指定的模式，并按固定顺序排序
    modes_to_run = [mode for mode in mode_order if mode in modes]

    # 按固定顺序执行所有指定的模式
    for i, mode in enumerate(modes_to_run):
        if mode in mode_handlers:
            if len(modes_to_run) > 1:
                logger.note(f"\n{'='*60}")
                logger.note(f"执行模式 [{i+1}/{len(modes_to_run)}]: {mode}")
                logger.note(f"{'='*60}\n")
            mode_handlers[mode](args, config)
            if i < len(modes_to_run) - 1:  # 不是最后一个模式
                print()  # 添加空行分隔
        else:
            logger.err(f"未知模式: {mode}")


if __name__ == "__main__":
    with Runtimer():
        main()

    # 使用示例:
    # Case: 导出：PyTorch -> ONNX -> TensorRT
    # python -m gtaz.agency_move.inference -m export
    # python -m gtaz.agency_move.inference -m export -w
    # python -m gtaz.agency_move.inference -m export -w -p path/to/model.pth
    # python -m gtaz.agency_move.inference -m export -w --no-fp16

    # Case: 推理：使用最新的 best 模型，默认 PyTorch 引擎
    # python -m gtaz.agency_move.inference -m infer
    # python -m gtaz.agency_move.inference -m infer --backend onnx
    # python -m gtaz.agency_move.inference -m infer --backend torch,onnx,tensorrt

    # Case: 性能测试
    # python -m gtaz.agency_move.inference -m benchmark
    # python -m gtaz.agency_move.inference -m benchmark --iterations 200 --batch-size 4

    # Case: 验证
    # python -m gtaz.agency_move.inference -m validate

    # Case: 全链路：导出 + 推理 + 验证 + 性能测试
    # python -m gtaz.agency_move.inference -m all

    # Case: 多模式 (逗号分隔，优先级: export > infer > validate > benchmark)
    # python -m gtaz.agency_move.inference -m export,infer
    # python -m gtaz.agency_move.inference -m infer,validate,benchmark
    # python -m gtaz.agency_move.inference -m export,infer,validate,benchmark
