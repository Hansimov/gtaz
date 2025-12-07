"""GTAV 场景识别 - 基于 ResNet 的楼层分类

使用 timm 库和预训练 ResNet 进行楼层分类 (floor 1/2/3)
"""

import random
import numpy as np
import shutil
import sys
import time

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field
from tclogger import TCLogger, logstr, add_fills
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

logger = TCLogger("FloorRecognizerV2", use_prefix=True, use_prefix_ms=True)

MODULE_DIR = Path(__file__).parent
IMAGES_DIR = MODULE_DIR / "images"
CACHE_DIR = MODULE_DIR / "cache"


def log_header(header: str):
    header_str = add_fills(header, "=", total_width=80)
    logger.hint(f"{header_str}", use_prefix=False)


def logstr_count(right: int, wrong: int, total: int) -> str:
    ratio = right / total if total > 0 else 0
    right_str = logstr.okay(right)
    if wrong > 0:
        wrong_str = logstr.warn(wrong)
    else:
        wrong_str = logstr.okay(wrong)
    total_str = logstr.mesg(total)
    ratio_str = f"{ratio:.1%}"
    return f"{ratio_str} ({right_str}/{wrong_str}/{total_str})"


@dataclass
class MatchResult:
    """匹配结果"""

    floor: int  # 匹配到的楼层 (1, 2, 3)
    confidence: float  # 置信度 (0.0 - 1.0)
    score: float  # 匹配分数
    details: dict = field(default_factory=dict)


class FloorDataset(Dataset):
    """楼层数据集"""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        """
        Args:
            samples: [(image_path, floor_label), ...] 其中 floor_label 是 0/1/2 (对应楼层 1/2/3)
            transform: 图像变换
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class FloorClassifier(nn.Module):
    """基于 ResNet 的楼层分类器"""

    def __init__(self, num_classes: int = 3, model_name: str = "resnet18"):
        super().__init__()
        # 使用 timm 加载预训练模型
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        """提取特征（用于 KNN 等）"""
        return self.backbone(x)


class FloorRecognizerV2:
    """基于 ResNet 的楼层识别器"""

    def __init__(
        self,
        floor_count: int = 3,
        images_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        model_name: str = "resnet18",
        device: Optional[str] = None,
    ):
        self.floor_count = floor_count
        self.images_dir = images_dir or IMAGES_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.model_path = self.cache_dir / "floor_classifier_resnet.pth"
        self.model_name = model_name

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.note(f"使用设备: {self.device}")

        # 模型
        self.model = FloorClassifier(num_classes=floor_count, model_name=model_name)
        self.model.to(self.device)

        # 数据变换（使用 timm 的默认配置）
        self.data_config = resolve_data_config({}, model=self.model.backbone)
        self.transform = create_transform(**self.data_config)

        # 推理用的简单变换（不做数据增强）
        self.eval_transform = create_transform(**self.data_config, is_training=False)

    def _collect_samples(self) -> list[tuple[Path, int]]:
        """收集所有训练样本"""
        samples = []
        locations_dir = self.cache_dir / "locations"

        for floor in range(1, self.floor_count + 1):
            label = floor - 1  # 转为 0-indexed
            for d in locations_dir.glob(f"*floor_{floor}_minimap"):
                if not d.is_dir():
                    continue
                for f in d.glob("*.jpg"):
                    samples.append((f, label))

        return samples

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-4,
        save: bool = True,
    ):
        """训练分类器"""
        samples = self._collect_samples()
        if not samples:
            logger.warn("没有找到训练样本")
            return

        # 统计各类别样本数
        class_counts = [0] * self.floor_count
        for _, label in samples:
            class_counts[label] += 1
        for floor in range(1, self.floor_count + 1):
            logger.note(f"楼层 {floor} 样本数: {class_counts[floor-1]}")

        # 计算类别权重（处理样本不平衡）
        total = sum(class_counts)
        class_weights = [total / (c + 1) for c in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        class_weights = class_weights / class_weights.sum() * self.floor_count

        # 创建数据增强变换
        train_transform = create_transform(
            **self.data_config,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
        )

        dataset = FloorDataset(samples, transform=train_transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # 冻结 backbone 的前几层
        for name, param in self.model.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 损失函数
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            right = 0
            total = 0

            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                right += predicted.eq(labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            acc = right / total if total > 0 else 0
            logger.mesg(
                f"Epoch {epoch+1}/{epochs}: Loss={total_loss:.4f}, Acc={acc:.1%}"
            )

        if save:
            self._save_model()

    def _save_model(self):
        """保存模型"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "floor_count": self.floor_count,
            },
            self.model_path,
        )
        logger.okay(f"模型已保存:")
        logger.okay(str(self.model_path))

    def load_model(self) -> bool:
        """加载模型"""
        if not self.model_path.exists():
            return False
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            logger.okay("模型已加载:")
            logger.okay(str(self.model_path))
            return True
        except Exception as e:
            logger.warn(f"加载失败: {e}")
            return False

    def recognize(self, image: Union[np.ndarray, Image.Image]) -> MatchResult:
        """识别楼层"""
        self.model.eval()

        # 转换图像
        if isinstance(image, np.ndarray):
            # BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            image = Image.fromarray(image)

        # 应用变换
        image_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1).squeeze()

        # 解析结果
        probs_np = probs.cpu().numpy()
        predicted_label = probs_np.argmax()
        predicted_floor = predicted_label + 1  # 转为 1-indexed
        confidence = probs_np[predicted_label]

        # 构建详情
        details = {f"floor_{i+1}": float(probs_np[i]) for i in range(self.floor_count)}

        return MatchResult(
            floor=int(predicted_floor),
            confidence=float(confidence),
            score=float(confidence),
            details=details,
        )

    def recognize_file(self, path: Union[str, Path]) -> Optional[MatchResult]:
        """从文件识别"""
        try:
            image = Image.open(path).convert("RGB")
            return self.recognize(image)
        except Exception as e:
            logger.warn(f"读取图像失败: {e}")
            return None

    def evaluate(
        self, test_dir: Path, expected_floor: int, collect_errors: bool = False
    ) -> dict:
        """评估准确率

        Args:
            test_dir: 测试目录
            expected_floor: 期望的楼层
            collect_errors: 是否收集错误分类的样本

        Returns:
            包含 accuracy, total, right, errors (如果 collect_errors=True) 的字典
        """
        files = list(Path(test_dir).glob("*.jpg"))
        if not files:
            return {"accuracy": 0, "total": 0, "right": 0, "errors": []}
        right = 0
        errors = []  # [(file_path, predicted_floor), ...]

        for f in files:
            r = self.recognize_file(f)
            if r:
                if r.floor == expected_floor:
                    right += 1
                elif collect_errors:
                    errors.append((f, r.floor))

        result = {
            "expected_floor": expected_floor,
            "total": len(files),
            "right": right,
            "accuracy": right / len(files) if files else 0,
        }
        if collect_errors:
            result["errors"] = errors
        return result

    def __repr__(self) -> str:
        return f"FloorRecognizerV2(model={self.model_name}, device={self.device})"


# ============================================================
# 测试
# ============================================================


def test_train():
    """训练模型"""
    log_header("训练模型")

    recognizer = FloorRecognizerV2()
    recognizer.train(epochs=25, batch_size=16, lr=1e-4)


def test_accuracy():
    """测试准确率"""
    log_header("准确率测试")

    recognizer = FloorRecognizerV2()
    if not recognizer.load_model():
        logger.warn("未找到模型，先训练...")
        recognizer.train(epochs=25)

    logger.note(f"{recognizer}")

    locations_dir = CACHE_DIR / "locations"
    bad_dir = CACHE_DIR / "bad_recognizes_v2"
    results = []
    all_errors = []

    for d in sorted(locations_dir.glob("*_minimap")):
        for floor in range(1, 4):
            if f"floor_{floor}" in d.name:
                r = recognizer.evaluate(d, floor, collect_errors=True)
                right = r["right"]
                total = r["total"]
                wrong = total - right
                results.append(r)
                logger.mesg(f"{d.name}: {logstr_count(right, wrong, total)}")
                # 收集错误样本，附带原始目录名
                if r.get("errors"):
                    for err_path, pred_floor in r["errors"]:
                        all_errors.append((err_path, pred_floor, d.name))
                break

    if results:
        total = sum(r["total"] for r in results)
        right = sum(r["right"] for r in results)
        wrong = total - right
        logger.okay(f"平均准确率: {logstr_count(right, wrong, total)}")

    # 复制错误分类的图片到 bad_recognizes_v2
    if all_errors:
        bad_dir.mkdir(parents=True, exist_ok=True)
        # 清空旧的错误文件
        for old_file in bad_dir.glob("*.jpg"):
            old_file.unlink()

        for err_path, pred_floor, folder_name in all_errors:
            folder_base = folder_name.replace("_minimap", "")
            # 新文件名: <original_folder_without_minimap>_as_<result_floor_class>_<original_filename>.jpg
            new_name = f"{folder_base}_as_{pred_floor}_{err_path.name}"
            new_path = bad_dir / new_name
            shutil.copy2(err_path, new_path)

        logger.okay(f"错误分类图片 {len(all_errors)} 张，已复制到:")
        logger.file(str(bad_dir))


def stratified_k_fold(
    samples: list[tuple[Path, int]], n_splits: int = 5, seed: int = 42
):
    """手动实现分层 K-Fold 交叉验证

    Args:
        samples: [(path, label), ...]
        n_splits: 折数
        seed: 随机种子

    Yields:
        (train_indices, val_indices) for each fold
    """
    random.seed(seed)

    # 按类别分组
    class_indices: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    # 打乱每个类别的索引
    for indices in class_indices.values():
        random.shuffle(indices)

    # 为每个 fold 分配样本
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    # 生成 train/val 划分
    for fold_idx in range(n_splits):
        val_indices = folds[fold_idx]
        train_indices = []
        for i, fold in enumerate(folds):
            if i != fold_idx:
                train_indices.extend(fold)
        yield train_indices, val_indices


def test_cross_validation():
    """K-Fold 交叉验证"""
    log_header("K-Fold 交叉验证")

    recognizer = FloorRecognizerV2()
    samples = recognizer._collect_samples()

    if not samples:
        logger.warn("没有找到样本")
        return

    # 5-Fold 交叉验证
    n_splits = 5
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        stratified_k_fold(samples, n_splits=n_splits)
    ):
        logger.note(f"Fold {fold + 1}/5")

        # 重新初始化模型
        recognizer = FloorRecognizerV2()

        # 准备数据
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        # 训练（只用训练集）
        train_transform = create_transform(
            **recognizer.data_config,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
        )
        dataset = FloorDataset(train_samples, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        # 类别权重
        class_counts = [0] * recognizer.floor_count
        for _, label in train_samples:
            class_counts[label] += 1
        class_weights = [len(train_samples) / (c + 1) for c in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
            recognizer.device
        )
        class_weights = class_weights / class_weights.sum() * recognizer.floor_count

        # 冻结部分层
        for name, param in recognizer.model.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, recognizer.model.parameters()),
            lr=1e-4,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 训练 15 epochs
        recognizer.model.train()
        for epoch in range(15):
            for images, labels_batch in dataloader:
                images = images.to(recognizer.device)
                labels_batch = labels_batch.to(recognizer.device)
                optimizer.zero_grad()
                outputs = recognizer.model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # 验证
        recognizer.model.eval()
        right = 0
        for path, label in val_samples:
            result = recognizer.recognize_file(path)
            if result and result.floor == label + 1:
                right += 1

        acc = right / len(val_samples)
        fold_results.append(acc)
        logger.mesg(f"Fold {fold + 1} Accuracy: {acc:.1%}")

    avg_acc = sum(fold_results) / len(fold_results)
    std_acc = (sum((x - avg_acc) ** 2 for x in fold_results) / len(fold_results)) ** 0.5
    logger.okay(f"平均准确率: {avg_acc:.1%} ± {std_acc:.1%}")


def test_speed():
    """测试速度"""
    log_header("速度测试")

    recognizer = FloorRecognizerV2()
    if not recognizer.load_model():
        logger.warn("未找到模型，先训练...")
        recognizer.train(epochs=25)

    # 收集测试图像
    test_files = []
    for d in (CACHE_DIR / "locations").glob("*_minimap"):
        test_files.extend(list(d.glob("*.jpg"))[:5])

    if not test_files:
        logger.warn("没有测试图像")
        return

    # 预热
    recognizer.recognize_file(test_files[0])

    start = time.time()
    for f in test_files:
        recognizer.recognize_file(f)
    elapsed = time.time() - start

    logger.okay(
        f"平均: {elapsed/len(test_files)*1000:.1f} ms/张 ({len(test_files)} 张)"
    )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "train":
            test_train()
        elif cmd == "test":
            test_accuracy()
        elif cmd == "cv":
            test_cross_validation()
        elif cmd == "speed":
            test_speed()
    else:
        test_train()
        test_accuracy()
        test_speed()

    # python -m gtaz.recognizes_v2           # 训练+测试+速度
    # python -m gtaz.recognizes_v2 train     # 仅训练
    # python -m gtaz.recognizes_v2 test      # 仅测试
    # python -m gtaz.recognizes_v2 cv        # 交叉验证
    # python -m gtaz.recognizes_v2 speed     # 速度测试
