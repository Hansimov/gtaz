"""
GTAV 行为克隆 (Behavior Cloning) 模型训练

基于 ResNet18 的端到端行为克隆模型，用于从小地图帧预测 WASD 按键动作。

架构设计（高内聚、低耦合）:
- Config: 统一配置管理
- FrameSample: 单帧数据结构
- DataProcessor: 数据解析和处理（静态方法）
- DataManager: 数据加载、序列创建和数据集划分
- AgencyMoveDataset: PyTorch 数据集封装
- TemporalResNet/SimpleCNNModel: 模型定义
- ModelFactory: 模型创建工厂
- Metrics: 评估指标计算
- Trainer: 训练流程管理
- Predictor: 推理预测
- AgencyMovePipeline: 顶层流程编排
"""

import argparse
import json
import random
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tclogger import TCLogger, TCLogbar, logstr, dict_to_str, Runtimer
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


logger = TCLogger(name="AgencyMove", use_prefix=True)


# ===== 常量定义 ===== #

KEY_TO_INDEX = {"W": 0, "A": 1, "S": 2, "D": 3}
INDEX_TO_KEY = {0: "W", 1: "A", 2: "S", 3: "D"}
NUM_KEYS = 4


# ===== 配置类 ===== #

# src/gtaz
SRC_DIR = Path(__file__).parent.parent
DATA_DIR = SRC_DIR / "cache/agency_move"
CKPT_DIR = SRC_DIR / "checkpoints/agency_move"


@dataclass
class Config:
    """统一配置管理"""

    # 数据相关
    data_dir: str = str(DATA_DIR)
    history_frames: int = 4  # 模型输入的历史参考帧数量
    image_size: tuple[int, int] = (160, 220)  # (H, W)
    use_key_history: bool = True
    min_gap_threshold: float = 1.0  # 帧间隔阈值（秒）
    use_image_cache: bool = True  # 是否使用图像缓存（提升性能但增加内存占用）

    # 模型相关
    backbone: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = 256
    dropout: float = 0.3
    model_type: str = "temporal"  # "temporal" or "simple"

    # 训练相关
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: Optional[float] = None  # None 表示使用自动调整
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    class_weights: tuple[float, float, float, float] = (1.0, 2.0, 2.0, 2.0)

    # 数据划分
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.1  # 不浪费训练和验证数据，随机选取一定比例测试

    # 其他
    # num_workers: int = 0 if platform.system() == "Windows" else 4
    num_workers: int = 4 if torch.cuda.is_available() else 2  # GPU可用时使用更多worker
    prefetch_factor: int = 4  # 预取因子，每个worker预取的batch数
    persistent_workers: bool = True  # 保持worker进程存活，避免重复创建
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = str(CKPT_DIR)
    log_interval: int = 10
    threshold: float = 0.5  # 预测阈值
    overwrite: bool = False  # 是否覆盖旧模型，不使用checkpoint恢复

    def to_dict(self) -> dict:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_model_name(self) -> str:
        """生成模型名称（基于训练参数）"""
        # 构建参数字符串
        params = [
            f"{self.model_type}",
            f"f{self.history_frames}",
            f"b{self.batch_size}",
            f"e{self.num_epochs}",
        ]

        # 添加学习率信息
        if self.learning_rate is None:
            params.append("lr_auto")
        else:
            lr_str = f"{self.learning_rate:.0e}".replace("e-0", "e-").replace(
                "e+0", "e"
            )
            params.append(f"lr{lr_str}")

        # 添加dropout信息（如果不是默认值）
        if self.dropout != 0.3:
            params.append(f"drop{self.dropout:.2f}".replace("0.", "."))

        # 添加hidden_dim信息（如果不是默认值）
        if self.hidden_dim != 256:
            params.append(f"h{self.hidden_dim}")

        return f"agency_move_{'_'.join(params)}"


# ===== 数据结构 ===== #


@dataclass
class FrameSample:
    """单帧数据样本"""

    image_path: str
    json_path: str
    frame_index: int
    timestamp: float
    keys_pressed: list[str] = field(default_factory=list)
    key_vector: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KEYS))


# ===== 数据处理器（静态方法集合） ===== #


class DataProcessor:
    """数据解析和处理器（静态工具类）"""

    @staticmethod
    def parse_filename(filename: str) -> tuple[str, int]:
        """解析文件名，提取时间戳和帧索引"""
        pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3})_(\d+)"
        match = re.match(pattern, filename)
        if match:
            return match.group(1), int(match.group(2))
        return "", 0

    @staticmethod
    def is_segment_start(filename: str) -> bool:
        """判断文件名是否标志着新段的开始（末尾为 _0001 或 _001 等）"""
        pattern = r"_0+1\.(json|jpg)$"
        return bool(re.search(pattern, filename))

    @staticmethod
    def load_json(json_path: str) -> dict:
        """加载 JSON 数据文件"""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def extract_key_vector(json_data: dict) -> np.ndarray:
        """从 JSON 数据中提取按键向量"""
        key_vector = np.zeros(NUM_KEYS, dtype=np.float32)
        if json_data.get("has_action", False) and "keys" in json_data:
            for key_info in json_data["keys"]:
                key_name = key_info.get("key_name", "")
                is_pressed = key_info.get("is_pressed", False)
                if key_name in KEY_TO_INDEX and is_pressed:
                    key_vector[KEY_TO_INDEX[key_name]] = 1.0
        return key_vector

    @classmethod
    def parse_frame(cls, json_path: Path) -> Optional[FrameSample]:
        """解析单帧数据"""
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            return None

        try:
            json_data = cls.load_json(str(json_path))
        except (json.JSONDecodeError, IOError):
            return None

        _, frame_index = cls.parse_filename(json_path.name)
        timestamp = json_data.get("time", {}).get("timestamp", 0)
        key_vector = cls.extract_key_vector(json_data)
        keys_pressed = [INDEX_TO_KEY[i] for i in range(NUM_KEYS) if key_vector[i] > 0]

        return FrameSample(
            image_path=str(image_path),
            json_path=str(json_path),
            frame_index=frame_index,
            timestamp=timestamp,
            keys_pressed=keys_pressed,
            key_vector=key_vector,
        )

    @classmethod
    def collect_session_frames(cls, session_dir: Path) -> list[list[FrameSample]]:
        """收集一个会话目录中的所有帧数据，按段分组"""
        json_files = sorted(session_dir.glob("*.json"))

        segments = []
        current_segment = []

        for json_path in json_files:
            # 如果是新段的开始且当前段不为空，保存当前段
            if cls.is_segment_start(json_path.name) and current_segment:
                segments.append(current_segment)
                current_segment = []

            sample = cls.parse_frame(json_path)
            if sample:
                current_segment.append(sample)

        # 保存最后一段
        if current_segment:
            segments.append(current_segment)

        # 按时间戳排序每个段内的样本
        for segment in segments:
            segment.sort(key=lambda x: x.timestamp)

        return segments


# ===== 数据集类 ===== #


class AgencyMoveDataset(Dataset):
    """GTAV 行为克隆数据集"""

    def __init__(
        self,
        sequences: list[list[FrameSample]],
        history_frames: int = 4,
        use_key_history: bool = True,
        image_size: tuple[int, int] = (160, 220),
        augment: bool = False,
        use_cache: bool = True,
    ):
        self.sequences = sequences
        self.history_frames = history_frames
        self.use_key_history = use_key_history
        self.image_size = image_size
        self.augment = augment
        self.use_cache = use_cache
        self.transform = self._build_transform()
        # 缓存预处理后的tensor而非PIL Image，减少CPU处理
        self._tensor_cache = {} if use_cache else None

    def _build_transform(self) -> transforms.Compose:
        """构建图像变换"""
        if self.augment:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1
                    ),
                    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def warmup_cache(self):
        """预热缓存：预先加载所有图像到缓存中（仅对验证/测试集）"""
        if not self.use_cache or self.augment:
            return

        # 收集所有唯一的图像路径
        unique_paths = set()
        for seq in self.sequences:
            for sample in seq:
                unique_paths.add(sample.image_path)

        if not unique_paths:
            return

        logger.note(f"预热缓存: 加载 {len(unique_paths)} 张图像...")
        warmup_desc = logstr.note("* 预热缓存")
        bar = TCLogbar(total=len(unique_paths), desc=warmup_desc)

        for img_path in unique_paths:
            if img_path not in self._tensor_cache:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
                self._tensor_cache[img_path] = img_tensor
            bar.update(1)

        bar.update(flush=True, linebreak=True)
        logger.okay(f"缓存预热完成: {len(self._tensor_cache)} 张图像")

    def _load_and_transform_image(self, image_path: str) -> torch.Tensor:
        """加载并转换图像为tensor，支持缓存"""
        # 对于验证/测试集，缓存预处理后的tensor
        if self.use_cache and not self.augment and image_path in self._tensor_cache:
            return self._tensor_cache[image_path]

        # 加载并转换图像
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)

        # 对于验证/测试集缓存tensor（训练集有数据增强所以不缓存）
        if self.use_cache and not self.augment:
            self._tensor_cache[image_path] = img_tensor

        return img_tensor

    def __getitem__(self, idx: int) -> dict:
        sequence = self.sequences[idx]

        # 加载并处理图像 - 直接获取tensor
        images = [self._load_and_transform_image(s.image_path) for s in sequence]
        images = torch.stack(images, dim=0)

        # 获取按键信息 - 预分配numpy数组提升效率
        seq_len = len(sequence)
        key_vectors = np.empty((seq_len, NUM_KEYS), dtype=np.float32)
        for i, s in enumerate(sequence):
            key_vectors[i] = s.key_vector

        key_history = torch.from_numpy(key_vectors)
        target = key_history[-1]

        result = {"images": images, "target": target}
        if self.use_key_history:
            result["key_history"] = key_history[:-1]

        return result


# ===== 数据管理器 ===== #


class DataManager:
    """数据管理器：负责数据加载、序列创建和数据集划分"""

    def __init__(self, config: Config):
        self.config = config
        self._sequences: list[list[FrameSample]] = []

    @property
    def sequences(self) -> list[list[FrameSample]]:
        return self._sequences

    def create_sequences(self, samples: list[FrameSample]) -> list[list[FrameSample]]:
        """将帧样本按历史帧数量分割成连续的序列"""
        history_frames = self.config.history_frames
        min_gap = self.config.min_gap_threshold

        if len(samples) < history_frames:
            return []

        sequences = []
        current_seq = [samples[0]]

        for i in range(1, len(samples)):
            time_gap = samples[i].timestamp - samples[i - 1].timestamp
            if time_gap > min_gap or time_gap < 0:
                current_seq = [samples[i]]
            else:
                current_seq.append(samples[i])

            if len(current_seq) >= history_frames:
                for j in range(len(current_seq) - history_frames + 1):
                    sequences.append(current_seq[j : j + history_frames])
                current_seq = current_seq[-(history_frames - 1) :]

        return sequences

    def load_all_data(self) -> list[list[FrameSample]]:
        """加载所有训练数据"""
        data_dir = Path(self.config.data_dir)
        all_sequences = []

        session_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        logger.note(f"找到 {logstr.mesg(len(session_dirs))} 个会话目录")

        total_segments = 0

        load_desc = logstr.note("* 加载数据")
        bar = TCLogbar(total=len(session_dirs), desc=load_desc)
        for session_dir in session_dirs:
            segments = DataProcessor.collect_session_frames(session_dir)
            total_segments += len(segments)

            # 对每个段分别创建序列
            for segment in segments:
                if len(segment) >= self.config.history_frames:
                    sequences = self.create_sequences(segment)
                    all_sequences.extend(sequences)
            bar.update(1)
        bar.update(flush=True, linebreak=True)

        self._sequences = all_sequences
        logger.okay(
            f"从 {logstr.mesg(total_segments)} 个数据段中"
            f"总共生成 {logstr.mesg(len(all_sequences))} 个训练样本"
        )
        return all_sequences

    def split_data(self) -> tuple[list, list, list]:
        """划分数据集"""
        if not self._sequences:
            raise ValueError("没有可用的序列数据！请先调用 load_all_data()")

        sequences = self._sequences.copy()
        random.shuffle(sequences)

        total = len(sequences)
        train_size = int(total * self.config.train_ratio)

        train_seqs = sequences[:train_size]
        val_seqs = sequences[train_size:]

        test_size = int(total * self.config.test_ratio)
        test_seqs = sequences[-test_size:]

        logger.note("> 数据集划分:")
        split_info = {
            "训练集": len(train_seqs),
            "验证集": len(val_seqs),
            "测试集": len(test_seqs),
        }
        logger.mesg(dict_to_str(split_info), indent=2)
        return train_seqs, val_seqs, test_seqs

    def create_datasets(
        self,
    ) -> tuple[AgencyMoveDataset, AgencyMoveDataset, AgencyMoveDataset]:
        """创建数据集"""
        train_seqs, val_seqs, test_seqs = self.split_data()

        train_ds = AgencyMoveDataset(
            train_seqs,
            self.config.history_frames,
            self.config.use_key_history,
            self.config.image_size,
            augment=True,
            use_cache=self.config.use_image_cache,
        )
        val_ds = AgencyMoveDataset(
            val_seqs,
            self.config.history_frames,
            self.config.use_key_history,
            self.config.image_size,
            augment=False,
            use_cache=self.config.use_image_cache,
        )
        test_ds = AgencyMoveDataset(
            test_seqs,
            self.config.history_frames,
            self.config.use_key_history,
            self.config.image_size,
            augment=False,
            use_cache=self.config.use_image_cache,
        )

        # 预热验证和测试集缓存
        if self.config.use_image_cache:
            val_ds.warmup_cache()
            test_ds.warmup_cache()

        return train_ds, val_ds, test_ds

    def create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        if not self._sequences:
            self.load_all_data()

        train_ds, val_ds, test_ds = self.create_datasets()

        # 优化的DataLoader配置
        common_kwargs = {
            "num_workers": self.config.num_workers,
            "pin_memory": True,
            "prefetch_factor": (
                self.config.prefetch_factor if self.config.num_workers > 0 else None
            ),
            "persistent_workers": (
                self.config.persistent_workers if self.config.num_workers > 0 else False
            ),
        }

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            **common_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            **common_kwargs,
        )

        return train_loader, val_loader, test_loader

    def analyze(self) -> dict:
        """分析数据集统计信息"""
        if not self._sequences:
            self.load_all_data()

        key_counts = {key: 0 for key in INDEX_TO_KEY.values()}
        key_combo_counts = {}
        total = len(self._sequences)

        for seq in self._sequences:
            sample = seq[-1]
            keys = tuple(sorted(sample.keys_pressed))
            key_combo_counts[keys] = key_combo_counts.get(keys, 0) + 1
            for key in sample.keys_pressed:
                key_counts[key] += 1

        # 打印统计信息
        logger.note(f"> 数据集统计: 总样本数 {logstr.mesg(total)}")

        logger.note("单键分布:")
        key_dist = {}
        for key, count in key_counts.items():
            ratio = f"{count / total * 100:.2f}%" if total > 0 else "0%"
            key_dist[key] = f"{count} ({ratio})"
        logger.mesg(dict_to_str(key_dist), indent=2)

        logger.note("按键组合分布 (Top 10):")
        combo_dist = {}
        for combo, count in sorted(key_combo_counts.items(), key=lambda x: -x[1])[:10]:
            combo_str = "+".join(combo) if combo else "(无按键)"
            ratio = f"{count / total * 100:.2f}%" if total > 0 else "0%"
            combo_dist[combo_str] = f"{count} ({ratio})"
        logger.mesg(dict_to_str(combo_dist), indent=2)

        return {
            "total_samples": total,
            "key_counts": key_counts,
            "key_combo_counts": key_combo_counts,
        }


# ===== 模型定义 ===== #


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


class SimpleCNNModel(nn.Module):
    """简化 CNN 模型：将多个历史帧图像在通道维度堆叠"""

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
        self.history_frames = history_frames
        self.num_keys = num_keys
        self.use_key_history = use_key_history

        # 修改 ResNet18 第一层
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            history_frames * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            with torch.no_grad():
                weight = (
                    original_conv.weight.data.repeat(1, history_frames, 1, 1)
                    / history_frames
                )
                self.conv1.weight.data = weight

        self.backbone = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        fc_input_dim = 512
        if use_key_history:
            self.key_embedding = nn.Sequential(
                nn.Linear((history_frames - 1) * num_keys, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fc_input_dim += 64
        else:
            self.key_embedding = None

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
        images = images.view(batch_size, num_frames_in_batch * C, H, W)

        x = self.conv1(images)
        x = self.backbone(x)
        features = x.view(batch_size, -1)

        if self.use_key_history and key_history is not None:
            key_flat = key_history.view(batch_size, -1)
            key_features = self.key_embedding(key_flat)
            combined = torch.cat([features, key_features], dim=1)
        else:
            combined = features

        return self.fc(combined)


# ===== 模型工厂 ===== #


class ModelFactory:
    """模型创建工厂"""

    @staticmethod
    def create(config: Config) -> nn.Module:
        """根据配置创建模型"""
        model_cls = (
            TemporalResNet if config.model_type == "temporal" else SimpleCNNModel
        )
        return model_cls(
            history_frames=config.history_frames,
            num_keys=NUM_KEYS,
            hidden_dim=config.hidden_dim,
            use_key_history=config.use_key_history,
            dropout=config.dropout,
            pretrained=config.pretrained,
        )

    @staticmethod
    def load(model_path: str, device: torch.device = None) -> tuple[nn.Module, dict]:
        """加载模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config_dict = checkpoint.get("config", {})

        model = TemporalResNet(
            history_frames=config_dict.get("history_frames", 4),
            num_keys=NUM_KEYS,
            hidden_dim=config_dict.get("hidden_dim", 256),
            use_key_history=config_dict.get("use_key_history", True),
            dropout=config_dict.get("dropout", 0.3),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        return model, config_dict


# ===== 评估指标 ===== #


class Metrics:
    """评估指标计算器"""

    def __init__(self, num_keys: int = NUM_KEYS, threshold: float = 0.5):
        self.num_keys = num_keys
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_keys)
        self.fp = np.zeros(self.num_keys)
        self.tn = np.zeros(self.num_keys)
        self.fn = np.zeros(self.num_keys)
        self.exact_match = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(predictions).detach().cpu().numpy()
        preds = (probs > self.threshold).astype(np.float32)
        targets_np = targets.detach().cpu().numpy()

        for i in range(self.num_keys):
            p, t = preds[:, i], targets_np[:, i]
            self.tp[i] += np.sum((p == 1) & (t == 1))
            self.fp[i] += np.sum((p == 1) & (t == 0))
            self.tn[i] += np.sum((p == 0) & (t == 0))
            self.fn[i] += np.sum((p == 0) & (t == 1))

        self.exact_match += np.sum(np.all(preds == targets_np, axis=1))
        self.total += len(targets_np)

    def compute(self) -> dict:
        results = {}
        for i, key_name in INDEX_TO_KEY.items():
            precision = self.tp[i] / (self.tp[i] + self.fp[i] + 1e-8)
            recall = self.tp[i] / (self.tp[i] + self.fn[i] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (self.tp[i] + self.tn[i]) / (
                self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i] + 1e-8
            )
            results.update(
                {
                    f"{key_name}_precision": precision,
                    f"{key_name}_recall": recall,
                    f"{key_name}_f1": f1,
                    f"{key_name}_accuracy": accuracy,
                }
            )

        results["avg_precision"] = np.mean(
            [results[f"{k}_precision"] for k in INDEX_TO_KEY.values()]
        )
        results["avg_recall"] = np.mean(
            [results[f"{k}_recall"] for k in INDEX_TO_KEY.values()]
        )
        results["avg_f1"] = np.mean([results[f"{k}_f1"] for k in INDEX_TO_KEY.values()])
        results["avg_accuracy"] = np.mean(
            [results[f"{k}_accuracy"] for k in INDEX_TO_KEY.values()]
        )
        results["exact_match_accuracy"] = self.exact_match / (self.total + 1e-8)

        return results


# ===== 训练器 ===== #


class Trainer:
    """训练器：管理模型训练流程"""

    def __init__(self, config: Config, model: nn.Module, device: torch.device = None):
        self.config = config
        self.model = model
        self.device = device or torch.device(config.device)
        self.model.to(self.device)

        # 混合精度训练（仅GPU可用）
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 编译模型以提升GPU利用率（PyTorch 2.0+，仅Linux）
        # Windows上torch.compile需要Triton但不可用，因此禁用
        self.model_compiled = False
        if (
            torch.cuda.is_available()
            and hasattr(torch, "compile")
            and platform.system() != "Windows"
        ):
            try:
                self.model = torch.compile(self.model, mode="default")
                self.model_compiled = True
                logger.note("模型编译成功")
            except Exception as e:
                logger.warn(f"模型编译失败，使用普通模式: {e}")
                self.model_compiled = False

        # 损失函数
        self.class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(
            self.device
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        # 优化器和学习率调度器
        if config.learning_rate is None:
            # 使用自动调整学习率：从较大的初始值开始
            initial_lr = 1e-4
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=initial_lr,
                weight_decay=config.weight_decay,
            )
            # 使用 ReduceLROnPlateau：根据验证指标自动调整
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # 监控 F1 分数（越高越好）
                factor=0.5,  # 学习率减半
                patience=5,  # 5 个 epoch 没有提升则降低
                min_lr=1e-6,  # 最小学习率
            )
            self.use_plateau_scheduler = True
            logger.note(f"使用自动学习率调整，初始学习率: {logstr.mesg(initial_lr)}")
        else:
            # 使用固定学习率
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            # 学习率调度器
            if config.lr_scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.num_epochs
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
                )
            self.use_plateau_scheduler = False

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_exact_match": [],
            "val_exact_match": [],
        }
        self.latest_val_f1 = 0.0
        self.latest_epoch = 0
        self.start_epoch = 0
        self.best_val_f1 = 0.0  # 跟踪历史最佳验证F1

        # 跟踪最佳指标（用于计算百分比变化）
        self.best_metrics = {
            "train_loss": float("inf"),
            "val_loss": float("inf"),
            "train_f1": 0.0,
            "val_f1": 0.0,
            "train_exact_match": 0.0,
            "val_exact_match": 0.0,
        }

    def _format_metric_change(
        self, current: float, best: float, is_loss: bool = False
    ) -> str:
        """格式化指标变化（百分比，相对于最佳值）

        Args:
            current: 当前值
            best: 最佳值
            is_loss: 是否为损失指标（True表示越小越好，False表示越大越好）
        """
        if best == 0 or (is_loss and best == float("inf")):
            return ""

        percent_diff = ((current - best) / abs(best)) * 100
        abs_percent_diff = abs(percent_diff)

        # 对于loss，负数表示改善；对于其他指标，正数表示改善
        is_better = (percent_diff < 0) if is_loss else (percent_diff > 0)

        if abs_percent_diff < 0.05:
            return f" ({logstr.mesg(f'≈ best')})"
        elif is_better:
            return f" ({logstr.okay(f'↑ {abs_percent_diff:.1f}%')})"
        else:
            return f" ({logstr.warn(f'↓ {abs_percent_diff:.1f}%')})"

    def _log_checkpoint_info(self, checkpoint_path: str):
        """记录checkpoint恢复信息"""
        if self.latest_epoch >= self.config.num_epochs:
            logger.okay(f"> 训练已完成")
        else:
            logger.note(f"> 从 checkpoint 恢复训练:")
        info_dict = {
            "路径": checkpoint_path,
            "起始轮次": self.start_epoch,
            "上次F1": f"{self.latest_val_f1:.4f} (Epoch {self.latest_epoch})",
            "最佳F1": f"{self.best_val_f1:.4f}",
            "学习率": f"{self.optimizer.param_groups[0]['lr']:.6f}",
            "混合精度": "开启" if self.use_amp else "关闭",
            "模型编译": "开启" if self.model_compiled else "关闭",
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: dict,
        val_loss: float,
        val_metrics: dict,
    ):
        """记录epoch训练和验证指标"""
        current_lr = self.optimizer.param_groups[0]["lr"]
        logger.mesg(f"学习率: {logstr.file(f'{current_lr:.6f}')}")

        # 计算训练指标的变化（相对于最佳值）
        train_loss_change = ""
        train_f1_change = ""
        train_exact_change = ""
        if epoch > 0:
            train_loss_change = self._format_metric_change(
                train_loss, self.best_metrics["train_loss"], is_loss=True
            )
            train_f1_change = self._format_metric_change(
                train_metrics["avg_f1"], self.best_metrics["train_f1"], is_loss=False
            )
            train_exact_change = self._format_metric_change(
                train_metrics["exact_match_accuracy"],
                self.best_metrics["train_exact_match"],
                is_loss=False,
            )

        # 计算验证指标的变化（相对于最佳值）
        val_loss_change = ""
        val_f1_change = ""
        val_exact_change = ""
        if epoch > 0:
            val_loss_change = self._format_metric_change(
                val_loss, self.best_metrics["val_loss"], is_loss=True
            )
            val_f1_change = self._format_metric_change(
                val_metrics["avg_f1"], self.best_metrics["val_f1"], is_loss=False
            )
            val_exact_change = self._format_metric_change(
                val_metrics["exact_match_accuracy"],
                self.best_metrics["val_exact_match"],
                is_loss=False,
            )

        # 打印训练指标
        logger.mesg(
            f"训练 - Loss: {logstr.file(f'{train_loss:.4f}')}{train_loss_change}, "
            f"Avg F1: {logstr.file(f'{train_metrics['avg_f1']:.4f}')}{train_f1_change}, "
            f"Exact Match: {logstr.file(f'{train_metrics['exact_match_accuracy']:.4f}')}{train_exact_change}"
        )

        # 打印验证指标
        logger.mesg(
            f"验证 - Loss: {logstr.file(f'{val_loss:.4f}')}{val_loss_change}, "
            f"Avg F1: {logstr.file(f'{val_metrics['avg_f1']:.4f}')}{val_f1_change}, "
            f"Exact Match: {logstr.file(f'{val_metrics['exact_match_accuracy']:.4f}')}{val_exact_change}"
        )

    def _format_key_f1_scores(self, metrics: dict, prefix: str = "") -> str:
        """格式化各按键的F1分数为单行字符串

        Args:
            metrics: 包含各键F1分数的指标字典
            prefix: 前缀文本（如"训练"、"验证"、"测试"）

        Returns:
            格式化的字符串，如 "训练: W: 0.9886, A: 0.8992, S: 0.7273, D: 0.8068"
        """
        key_scores = []
        for key_name in INDEX_TO_KEY.values():
            f1_score = metrics[f"{key_name}_f1"]
            key_str = logstr.note(key_name)
            score_str = logstr.hint(f"{f1_score:.4f}")
            key_scores.append(f"{key_str}: {score_str}")
        scores_str = ", ".join(key_scores)
        if prefix:
            return f"{prefix}: {scores_str}"
        return scores_str

    def _log_per_key_metrics(self, train_metrics: dict, val_metrics: dict):
        """记录各按键的F1分数"""
        train_line = self._format_key_f1_scores(train_metrics, "训练")
        val_line = self._format_key_f1_scores(val_metrics, "验证")
        logger.mesg(train_line)
        logger.mesg(val_line)

    def _log_model_save(self, save_path: Path):
        """记录模型保存信息"""
        logger.okay(f"保存最新模型:")
        logger.file(f"{save_path}")

    def _save_latest_checkpoint(
        self, save_dir: Path, epoch: int, val_f1: float, val_loss: float
    ):
        """保存最新模型检查点"""
        model_name = self.config.get_model_name()
        save_path = save_dir / f"{model_name}.pth"
        self._save_checkpoint(save_path, epoch, val_f1, val_loss, is_best=False)
        self._log_model_save(save_path)

    def _save_best_checkpoint(
        self, save_dir: Path, epoch: int, val_f1: float, val_loss: float
    ):
        """保存最佳模型检查点（仅当F1提升时）"""
        model_name = self.config.get_model_name()
        best_save_path = save_dir / f"{model_name}_best.pth"
        self._save_checkpoint(best_save_path, epoch, val_f1, val_loss, is_best=True)
        logger.okay(f"保存最佳模型 (F1: {logstr.mesg(f'{self.best_val_f1:.4f}')})")
        logger.file(f"{best_save_path}")

    def _log_training_config(self):
        """记录训练配置信息"""
        logger.note("> 训练配置:")
        params_count = sum(p.numel() for p in self.model.parameters())
        params_count_str = f"{params_count/1e6:.1f}M"
        train_info = {
            "设备": str(self.device),
            "参数量": params_count_str,
            "混合精度": "开启" if self.use_amp else "关闭",
            "模型编译": "开启" if self.model_compiled else "关闭",
            "批次大小": self.config.batch_size,
            "工作进程": self.config.num_workers,
            "预取因子": self.config.prefetch_factor,
        }
        if self.start_epoch > 0 and self.start_epoch < self.config.num_epochs:
            train_info["状态"] = logstr.mesg(
                f"继续训练 (从 Epoch {self.start_epoch + 1} 开始)"
            )
        elif self.start_epoch >= self.config.num_epochs:
            train_info["状态"] = logstr.okay(f"训练已完成")
        else:
            train_info["状态"] = logstr.mesg(f"从头开始训练")
        logger.mesg(dict_to_str(train_info), indent=2)
        print()

    def _log_training_summary(self):
        """记录训练总结信息"""
        # 找到验证集最佳epoch
        best_val_f1_idx = self.history["val_f1"].index(max(self.history["val_f1"]))
        best_epoch = best_val_f1_idx + 1
        best_val_f1 = self.history["val_f1"][best_val_f1_idx]
        best_val_loss = self.history["val_loss"][best_val_f1_idx]
        best_val_exact = self.history["val_exact_match"][best_val_f1_idx]

        logger.okay("> 训练完成！")
        sep_str = f"\n{'=' * 70}\n"
        logger.okay(
            f"{sep_str}"
            f"最佳验证结果 (Epoch {logstr.mesg(best_epoch)}): "
            f"Loss={logstr.file(f'{best_val_loss:.4f}')}, "
            f"F1={logstr.mesg(f'{best_val_f1:.4f}')}, "
            f"Exact Match={logstr.file(f'{best_val_exact:.4f}')}"
            f"{sep_str}",
        )

    def _save_training_history(self, save_dir: Path):
        """保存训练历史到文件"""
        model_name = self.config.get_model_name()
        history_path = save_dir / f"{model_name}.history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.okay(f"保存训练历史:")
        logger.file(f"{history_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """从checkpoint恢复训练状态"""
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 恢复调度器状态（如果存在）
            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # 恢复训练历史
            if "history" in checkpoint:
                self.history = checkpoint["history"]

                # 从历史记录中恢复最佳指标
                for key in ["train_loss", "val_loss"]:
                    if self.history[key]:
                        self.best_metrics[key] = min(self.history[key])

                for key in [
                    "train_f1",
                    "val_f1",
                    "train_exact_match",
                    "val_exact_match",
                ]:
                    if self.history[key]:
                        self.best_metrics[key] = max(self.history[key])

            # 恢复最新指标
            self.latest_val_f1 = checkpoint.get(
                "latest_val_f1", checkpoint.get("val_f1", 0.0)
            )
            self.latest_epoch = checkpoint.get(
                "latest_epoch", checkpoint.get("epoch", 0)
            )
            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)

            self._log_checkpoint_info(checkpoint_path)
            return True
        except Exception as e:
            logger.fail(f"加载 checkpoint 失败: {e}")
            return False

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, dict]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        metrics = Metrics()
        num_batches = len(train_loader)

        train_desc = logstr.note("* Training  ")
        bar = TCLogbar(total=num_batches, desc=train_desc)

        for batch_idx, batch in enumerate(train_loader):
            # 使用非阻塞数据传输
            images = batch["images"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            key_history = batch.get("key_history")
            if key_history is not None:
                key_history = key_history.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度训练
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images, key_history)
                    loss = self.criterion(logits, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images, key_history)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

            # 减少.item()调用频率，累积loss
            total_loss += loss.detach()
            metrics.update(logits.detach(), targets)
            bar.update(1)

        bar.update(flush=True, linebreak=True)

        return (total_loss / num_batches).item(), metrics.compute()

    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        metrics = Metrics()

        val_desc = logstr.note("* Validation")
        bar = TCLogbar(total=len(val_loader), desc=val_desc)
        with torch.no_grad():
            for batch in val_loader:
                # 使用非阻塞数据传输
                images = batch["images"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                key_history = batch.get("key_history")
                if key_history is not None:
                    key_history = key_history.to(self.device, non_blocking=True)

                # 混合精度推理
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images, key_history)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(images, key_history)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                metrics.update(logits, targets)
                bar.update(1)
        bar.update(flush=True, linebreak=True)

        return total_loss / len(val_loader), metrics.compute()

    def _run_training_loop(
        self, train_loader: DataLoader, val_loader: DataLoader, save_dir: Path
    ):
        """运行训练循环"""
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self._current_epoch = epoch  # 记录当前epoch，用于中断时显示
            epoch_str = f"Epoch {logstr.mesg(epoch + 1)}/{self.config.num_epochs}"
            logger.note(f"{'='*30} [{epoch_str}] {'='*30}")

            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            # 调整学习率
            if self.use_plateau_scheduler:
                self.scheduler.step(val_metrics["avg_f1"])
            else:
                self.scheduler.step()

            # 打印结果（使用旧的 best_metrics 进行比较）
            self._log_epoch_metrics(
                epoch, train_loss, train_metrics, val_loss, val_metrics
            )

            # 更新最佳指标（在打印之后更新）
            self.best_metrics["train_loss"] = min(
                train_loss, self.best_metrics["train_loss"]
            )
            self.best_metrics["val_loss"] = min(val_loss, self.best_metrics["val_loss"])
            self.best_metrics["train_f1"] = max(
                train_metrics["avg_f1"], self.best_metrics["train_f1"]
            )
            self.best_metrics["val_f1"] = max(
                val_metrics["avg_f1"], self.best_metrics["val_f1"]
            )
            self.best_metrics["train_exact_match"] = max(
                train_metrics["exact_match_accuracy"],
                self.best_metrics["train_exact_match"],
            )
            self.best_metrics["val_exact_match"] = max(
                val_metrics["exact_match_accuracy"],
                self.best_metrics["val_exact_match"],
            )

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_metrics["avg_f1"])
            self.history["val_f1"].append(val_metrics["avg_f1"])
            self.history["train_exact_match"].append(
                train_metrics["exact_match_accuracy"]
            )
            self.history["val_exact_match"].append(val_metrics["exact_match_accuracy"])
            self._log_per_key_metrics(train_metrics, val_metrics)

            # 保存最新模型（每个epoch都保存）
            self.latest_val_f1 = val_metrics["avg_f1"]
            self.latest_epoch = epoch + 1
            self._save_latest_checkpoint(
                save_dir, epoch + 1, val_metrics["avg_f1"], val_loss
            )

            # 保存最佳模型（第1个epoch之后，仅当F1提升时保存）
            if epoch == 0:
                # 第1个epoch，初始化best_val_f1但不保存
                self.best_val_f1 = val_metrics["avg_f1"]
            elif val_metrics["avg_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["avg_f1"]
                self._save_best_checkpoint(
                    save_dir, epoch + 1, val_metrics["avg_f1"], val_loss
                )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> dict:
        """完整训练流程"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 加载checkpoint（如果指定）
        if resume_from:
            if not self.load_checkpoint(resume_from):
                logger.warn("无法加载 checkpoint，从头开始训练")
                self.start_epoch = 0

        # 记录训练配置
        self._log_training_config()

        # 运行训练循环
        self._run_training_loop(train_loader, val_loader, save_dir)

        # 记录训练总结
        self._log_training_summary()

        # 保存训练历史
        # self._save_training_history(save_dir)

        return self.history

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_f1: float,
        val_loss: float,
        is_best: bool = False,
    ):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_f1": val_f1,
            "val_loss": val_loss,
            "latest_val_f1": self.latest_val_f1,
            "latest_epoch": self.latest_epoch,
            "best_val_f1": self.best_val_f1,
            "history": self.history,
            "config": self.config.to_dict(),
            "is_best": is_best,
        }
        # 保存调度器状态
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        # 保存同名 JSON 文件记录 checkpoint 信息
        json_path = path.with_suffix(".json")
        checkpoint_info = {
            "checkpoint_path": str(path),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "val_f1": float(val_f1),
            "val_loss": float(val_loss),
            "latest_val_f1": float(self.latest_val_f1),
            "latest_epoch": self.latest_epoch,
            "best_val_f1": float(self.best_val_f1),
            "is_best": is_best,
            "current_lr": self.optimizer.param_groups[0]["lr"],
            "config": self.config.to_dict(),
            "history": self.history,
            "model_params": sum(p.numel() for p in self.model.parameters()),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)


# ===== 预测器 ===== #


class Predictor:
    """推理预测器"""

    def __init__(
        self, model: nn.Module, config: Config = None, device: torch.device = None
    ):
        self.model = model
        self.config = config or Config()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        images: torch.Tensor,
        key_history: Optional[torch.Tensor] = None,
        threshold: float = None,
    ) -> list[dict]:
        """预测按键"""
        threshold = threshold or self.config.threshold

        # 确保输入是 5D
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if key_history is not None and key_history.dim() == 2:
            key_history = key_history.unsqueeze(0)

        images = images.to(self.device)
        if key_history is not None:
            key_history = key_history.to(self.device)

        with torch.no_grad():
            logits = self.model(images, key_history)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        probs_np = probs.cpu().numpy()
        preds_np = self._postprocess(preds.cpu().numpy(), probs_np)

        results = []
        for i in range(len(preds_np)):
            results.append(
                {
                    "keys_pressed": [
                        INDEX_TO_KEY[j] for j in range(NUM_KEYS) if preds_np[i, j] > 0
                    ],
                    "probabilities": {
                        INDEX_TO_KEY[j]: float(probs_np[i, j]) for j in range(NUM_KEYS)
                    },
                    "raw_predictions": preds_np[i].tolist(),
                }
            )

        return results[0] if len(results) == 1 else results

    def _postprocess(self, preds: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """后处理：解决 W+S 或 A+D 冲突"""
        preds = preds.copy()
        for i in range(len(preds)):
            if preds[i, 0] > 0 and preds[i, 2] > 0:  # W+S
                preds[i, 2 if probs[i, 0] >= probs[i, 2] else 0] = 0
            if preds[i, 1] > 0 and preds[i, 3] > 0:  # A+D
                preds[i, 3 if probs[i, 1] >= probs[i, 3] else 1] = 0
        return preds


# ===== 顶层流程编排 ===== #


class AgencyMovePipeline:
    """顶层流程编排：整合所有组件"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._setup_seed()

    def _setup_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    @staticmethod
    def find_latest_model(save_dir: str) -> Optional[str]:
        """查找最新的模型文件"""
        save_path = Path(save_dir)
        if not save_path.exists():
            return None

        # 查找所有 .pth 文件
        model_files = list(save_path.glob("agency_move_*.pth"))
        if not model_files:
            return None

        # 按修改时间排序，返回最新的
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest_model)

    @staticmethod
    def find_checkpoint_for_config(config: Config) -> Optional[str]:
        """根据配置查找匹配的checkpoint文件"""
        save_path = Path(config.save_dir)
        if not save_path.exists():
            return None

        # 生成当前配置的模型名称
        model_name = config.get_model_name()
        checkpoint_path = save_path / f"{model_name}.pth"

        if checkpoint_path.exists():
            return str(checkpoint_path)
        return None

    def train(self) -> tuple[nn.Module, dict]:
        """执行训练流程"""
        # 查找匹配的checkpoint
        resume_from = None
        if not self.config.overwrite:
            resume_from = self.find_checkpoint_for_config(self.config)
            if resume_from:
                logger.note(f"已有 checkpoint: {logstr.file(resume_from)}")
            else:
                logger.note(f"没有 checkpoint，将从头开始训练")
        else:
            logger.note("覆盖模式：忽略已有 checkpoint，从头开始训练")

        # 数据准备
        data_manager = DataManager(self.config)
        data_manager.load_all_data()
        train_loader, val_loader, test_loader = data_manager.create_data_loaders()

        # 模型创建
        model = ModelFactory.create(self.config)

        # 训练
        trainer = Trainer(self.config, model)
        history = trainer.train(train_loader, val_loader, resume_from=resume_from)

        # 测试集评估
        logger.note("在测试集上评估...")
        test_loss, test_metrics = trainer.validate(test_loader)

        logger.okay("> 测试集结果:")
        test_results = {
            "Loss": f"{test_loss:.4f}",
            "平均F1": f"{test_metrics['avg_f1']:.4f}",
            "完全匹配": f"{test_metrics['exact_match_accuracy']:.4f}",
        }
        logger.mesg(dict_to_str(test_results), indent=2)

        test_line = trainer._format_key_f1_scores(test_metrics, "测试")
        logger.mesg(test_line)

        return model, history

    def analyze(self) -> dict:
        """分析数据集"""
        data_manager = DataManager(self.config)
        return data_manager.analyze()

    def test(self, model_path: Optional[str] = None) -> dict:
        """测试模型"""
        # 如果未指定模型，查找最新的模型
        if model_path is None:
            model_path = self.find_latest_model(self.config.save_dir)
            if model_path is None:
                logger.fail(f"在 {self.config.save_dir} 中找不到任何模型文件")
                raise FileNotFoundError("没有可用的模型文件")
            logger.note(f"使用最新模型: {logstr.file(model_path)}")

        # 加载模型
        model, _ = ModelFactory.load(model_path, torch.device(self.config.device))

        # 准备数据
        data_manager = DataManager(self.config)
        data_manager.load_all_data()
        _, _, test_loader = data_manager.create_data_loaders()

        # 评估
        trainer = Trainer(self.config, model)
        test_loss, test_metrics = trainer.validate(test_loader)

        logger.okay("> 测试集结果:")
        test_results = {
            "Loss": f"{test_loss:.4f}",
            "平均F1": f"{test_metrics['avg_f1']:.4f}",
            "完全匹配": f"{test_metrics['exact_match_accuracy']:.4f}",
        }
        logger.mesg(dict_to_str(test_results), indent=2)

        test_line = trainer._format_key_f1_scores(test_metrics, "测试")
        logger.mesg(test_line)

        return test_metrics


# ===== 参数解析器 ===== #


class TrainerArgParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        """创建参数解析器"""
        parser = argparse.ArgumentParser(description="GTAV 行为克隆模型训练")
        parser.add_argument(
            "-m",
            "--mode",
            type=str,
            default="train",
            choices=["train", "analyze", "test"],
            help="运行模式: train(训练), analyze(分析数据), test(测试模型)",
        )
        parser.add_argument(
            "-d",
            "--data-dir",
            type=str,
            default=str(DATA_DIR),
            help="数据目录路径",
        )
        parser.add_argument(
            "-p",
            "--model-path",
            type=str,
            default=None,
            help="模型文件路径（测试模式需要）",
        )
        parser.add_argument(
            "-n",
            "--history-frames",
            type=int,
            default=4,
            help="模型输入的历史参考帧数量",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=32,
            help="批次大小",
        )
        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=50,
            help="训练轮数",
        )
        parser.add_argument(
            "-l",
            "--lr",
            type=float,
            default=None,
            help="学习率（默认为None，使用自动调整）",
        )
        parser.add_argument(
            "-t",
            "--model-type",
            type=str,
            default="temporal",
            choices=["temporal", "simple"],
            help="模型类型: temporal(时序模型), simple(简单CNN)",
        )
        parser.add_argument(
            "-w",
            "--overwrite",
            action="store_true",
            help="覆盖模式：不加载已有 checkpoint，从头开始训练",
        )
        return parser

    def parse_args(self):
        """解析命令行参数"""
        return self.parser.parse_args()

    def create_config(self, args) -> Config:
        """从命令行参数创建配置对象"""
        return Config(
            data_dir=args.data_dir,
            history_frames=args.history_frames,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_type=args.model_type,
            overwrite=args.overwrite,
        )


# ===== 主函数 ===== #


def main():
    """主函数"""
    arg_parser = TrainerArgParser()
    args = arg_parser.parse_args()
    config = arg_parser.create_config(args)

    pipeline = AgencyMovePipeline(config)

    try:
        if args.mode == "train":
            pipeline.train()
        elif args.mode == "analyze":
            pipeline.analyze()
        elif args.mode == "test":
            pipeline.test(args.model_path)
    except KeyboardInterrupt:
        print()
        logger.warn("用户中断操作 (Ctrl+C)，退出程序")


if __name__ == "__main__":
    with Runtimer():
        main()

    # Case: 训练模式：50个epoch，批次大小32
    # python -m gtaz.agency_move.train -m train -e 50 -b 32
    # python -m gtaz.agency_move.train -m train -e 50 -b 64 -w

    # Case: 分析模式：分析数据集
    # python -m gtaz.agency_move.train -m analyze

    # Case: 测试模式：使用最新模型进行测试
    # python -m gtaz.agency_move.train -m test
