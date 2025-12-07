"""GTAV 场景识别 - 基于小地图的楼层识别

核心思路：使用简单有效的特征 + KNN分类
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from tclogger import TCLogger
import pickle

logger = TCLogger("FloorRecognizer")

MODULE_DIR = Path(__file__).parent.parent
IMAGES_DIR = MODULE_DIR / "images"
CACHE_DIR = MODULE_DIR / "cache"


@dataclass
class MatchResult:
    """匹配结果"""

    floor: int  # 匹配到的楼层 (1, 2, 3)
    confidence: float  # 置信度 (0.0 - 1.0)
    score: float  # 匹配分数
    details: dict = field(default_factory=dict)


class FloorRecognizer:
    """基于小地图的楼层识别器。"""

    def __init__(
        self,
        floor_count: int = 3,
        images_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.floor_count = floor_count
        self.images_dir = images_dir or IMAGES_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.feature_db_path = self.cache_dir / "floor_features.pkl"

        # 各楼层的样本特征库 {floor: [(features, weight), ...]}
        self.floor_samples: dict[int, list[tuple[np.ndarray, float]]] = {
            i: [] for i in range(1, floor_count + 1)
        }

        # 加载地图特征
        self._load_floor_maps()

    def _load_floor_maps(self):
        """加载各楼层的完整平面图特征。"""
        for floor in range(1, self.floor_count + 1):
            map_path = self.images_dir / f"floor_{floor}_minimap.png"
            if not map_path.exists():
                map_path = self.images_dir / f"floor_{floor}_fullmap.png"
            if not map_path.exists():
                continue

            floor_map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
            if floor_map is not None:
                features = self._extract_features(floor_map)
                # 地图特征权重较低（因为和实际小地图差异大）
                self.floor_samples[floor].append((features, 0.5))
                logger.okay(f"楼层 {floor} 地图加载成功")

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """提取用于楼层区分的特征。

        混合策略：局部特征（区分度高）+ 全局特征（泛化性好）
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 归一化尺寸
        image = cv2.resize(image, (64, 64))

        features = []

        # === 全局统计特征（泛化性好）===
        # 1. 全局灰度统计 (3)
        features.append(np.mean(image) / 255.0)
        features.append(np.std(image) / 128.0)
        features.append(np.median(image) / 255.0)

        # 2. 灰度直方图 (16 bins)
        hist = cv2.calcHist([image], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-10)
        features.extend(hist)

        # === 边缘特征（结构性）===
        edges = cv2.Canny(image, 50, 150)

        # 3. 全局边缘密度 (1)
        features.append(np.mean(edges > 0))

        # 4. 边缘方向分布 (4) - 不同楼层可能有不同的建筑结构
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # 水平边缘强度、垂直边缘强度
        features.append(np.mean(np.abs(sobelx)) / 255.0)
        features.append(np.mean(np.abs(sobely)) / 255.0)
        # 边缘强度标准差
        features.append(np.std(magnitude) / 255.0)

        # === 空间分布特征（3x3=9块，平衡局部和全局）===
        # 5. 分块均值 (9)
        block_size = 64 // 3
        for i in range(3):
            for j in range(3):
                y1, y2 = i * block_size, min((i + 1) * block_size, 64)
                x1, x2 = j * block_size, min((j + 1) * block_size, 64)
                block = image[y1:y2, x1:x2]
                features.append(np.mean(block) / 255.0)

        # 6. 分块边缘密度 (9)
        for i in range(3):
            for j in range(3):
                y1, y2 = i * block_size, min((i + 1) * block_size, 64)
                x1, x2 = j * block_size, min((j + 1) * block_size, 64)
                block = edges[y1:y2, x1:x2]
                features.append(np.mean(block > 0))

        # === 纹理特征 ===
        # 7. 拉普拉斯纹理复杂度 (1)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features.append(np.std(laplacian) / 100.0)

        # 8. 中心区域特征 (2) - 小地图中心通常是角色位置
        center = image[20:44, 20:44]
        features.append(np.mean(center) / 255.0)
        features.append(np.std(center) / 128.0)

        return np.array(features, dtype=np.float32)

    def _compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算欧氏距离。"""
        return np.linalg.norm(feat1 - feat2)

    def build_feature_database(self, save: bool = True):
        """从样本目录构建特征数据库。"""
        locations_dir = self.cache_dir / "locations"

        for floor in range(1, self.floor_count + 1):
            for d in locations_dir.glob(f"*floor_{floor}_minimap"):
                if not d.is_dir():
                    continue
                for file in d.glob("*.jpg"):
                    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # 原始特征
                        features = self._extract_features(image)
                        self.floor_samples[floor].append((features, 1.0))

            count = len(self.floor_samples[floor])
            logger.note(f"楼层 {floor} 样本数：{count}")

        if save:
            self._save_database()

    def _save_database(self):
        """保存特征库。"""
        data = {
            floor: [(f.tolist(), w) for f, w in samples]
            for floor, samples in self.floor_samples.items()
        }
        self.feature_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feature_db_path, "wb") as f:
            pickle.dump(data, f)
        logger.okay(f"特征库已保存: {self.feature_db_path}")

    def _load_database(self) -> bool:
        """加载特征库。"""
        if not self.feature_db_path.exists():
            return False
        try:
            with open(self.feature_db_path, "rb") as f:
                data = pickle.load(f)
            self.floor_samples = {
                int(k): [(np.array(f, dtype=np.float32), w) for f, w in v]
                for k, v in data.items()
            }
            logger.okay(f"特征库已加载")
            return True
        except Exception as e:
            logger.warn(f"加载失败: {e}")
            return False

    def recognize(self, image: np.ndarray) -> MatchResult:
        """识别楼层。使用加权KNN。"""
        features = self._extract_features(image)

        # 计算到所有样本的距离，按楼层分组
        floor_distances = {i: [] for i in range(1, self.floor_count + 1)}
        for floor, samples in self.floor_samples.items():
            for feat, weight in samples:
                dist = self._compute_distance(features, feat)
                floor_distances[floor].append((dist, weight))

        if all(len(d) == 0 for d in floor_distances.values()):
            return MatchResult(floor=1, confidence=0.0, score=0.0)

        # 对每个楼层排序，取每个楼层最近的k个样本
        k_per_floor = 3
        floor_scores = {i: 0.0 for i in range(1, self.floor_count + 1)}

        for floor, distances in floor_distances.items():
            if not distances:
                continue
            distances.sort(key=lambda x: x[0])
            top_k = distances[:k_per_floor]

            # 计算该楼层的得分（最近邻的距离越小，得分越高）
            for dist, weight in top_k:
                # 使用倒数权重
                dist_weight = 1.0 / (dist + 0.01)
                floor_scores[floor] += dist_weight * weight

        # 选择最佳
        best_floor = max(floor_scores.keys(), key=lambda f: floor_scores[f])
        best_score = floor_scores[best_floor]

        # 置信度
        sorted_scores = sorted(floor_scores.values(), reverse=True)
        if sorted_scores[0] > 0:
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            confidence = 0.0

        return MatchResult(
            floor=best_floor,
            confidence=confidence,
            score=best_score,
            details=floor_scores,
        )

    def recognize_file(self, path: Union[str, Path]) -> Optional[MatchResult]:
        """从文件识别。"""
        image = cv2.imread(str(path))
        if image is None:
            return None
        return self.recognize(image)

    def evaluate(self, test_dir: Path, expected_floor: int) -> dict:
        """评估准确率。"""
        files = list(Path(test_dir).glob("*.jpg"))
        if not files:
            return {"accuracy": 0, "total": 0, "correct": 0}

        correct = sum(
            1
            for f in files
            if (r := self.recognize_file(f)) and r.floor == expected_floor
        )
        return {
            "expected_floor": expected_floor,
            "total": len(files),
            "correct": correct,
            "accuracy": correct / len(files),
        }

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self.floor_samples.items()}
        return f"FloorRecognizer(samples={counts})"


# ============================================================
# 测试
# ============================================================


def test_cross_validation():
    """留一法交叉验证 - 测试真实泛化能力。

    每次测试一个样本时，从特征库中排除该样本，
    这样可以避免"用训练数据测试自己"的过拟合假象。
    """
    logger.note("=" * 60)
    logger.note("留一法交叉验证（真实泛化能力测试）")
    logger.note("=" * 60)

    # 创建一个识别器用于特征提取（只创建一次）
    base_recognizer = FloorRecognizer()

    # 收集所有样本的特征
    locations_dir = CACHE_DIR / "locations"
    all_samples: dict[int, list[tuple[Path, np.ndarray]]] = {1: [], 2: [], 3: []}

    for floor in range(1, 4):
        for d in locations_dir.glob(f"*floor_{floor}_minimap"):
            if not d.is_dir():
                continue
            for f in d.glob("*.jpg"):
                image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    features = base_recognizer._extract_features(image)
                    all_samples[floor].append((f, features))

    for floor, samples in all_samples.items():
        logger.note(f"楼层 {floor} 样本数：{len(samples)}")

    # 留一法交叉验证
    results = {
        1: {"correct": 0, "total": 0},
        2: {"correct": 0, "total": 0},
        3: {"correct": 0, "total": 0},
    }
    total_correct = 0
    total_samples = 0

    for test_floor in range(1, 4):
        for i, (test_file, test_features) in enumerate(all_samples[test_floor]):
            # 清空样本库，重新构建（排除当前测试样本）
            base_recognizer.floor_samples = {f: [] for f in range(1, 4)}

            # 添加所有其他样本
            for floor in range(1, 4):
                for j, (f, features) in enumerate(all_samples[floor]):
                    if floor == test_floor and j == i:
                        continue  # 排除测试样本
                    base_recognizer.floor_samples[floor].append((features, 1.0))

            # 直接用已提取的特征进行识别（避免重复读取图像）
            # 模拟 recognize 的逻辑
            all_distances = []
            # 按楼层分组计算距离
            floor_distances = {f: [] for f in range(1, 4)}
            for floor, samples in base_recognizer.floor_samples.items():
                for feat, weight in samples:
                    dist = base_recognizer._compute_distance(test_features, feat)
                    floor_distances[floor].append((dist, weight))

            # 每个楼层取最近的k个样本
            k_per_floor = 3
            floor_scores = {f: 0.0 for f in range(1, 4)}

            for floor, distances in floor_distances.items():
                if not distances:
                    continue
                distances.sort(key=lambda x: x[0])
                top_k = distances[:k_per_floor]
                for dist, weight in top_k:
                    dist_weight = 1.0 / (dist + 0.01)
                    floor_scores[floor] += dist_weight * weight

            predicted_floor = max(floor_scores.keys(), key=lambda f: floor_scores[f])

            results[test_floor]["total"] += 1
            if predicted_floor == test_floor:
                results[test_floor]["correct"] += 1

        # 每完成一个楼层输出进度
        r = results[test_floor]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        logger.mesg(f"楼层 {test_floor}: {acc:.1%} ({r['correct']}/{r['total']})")
        total_correct += r["correct"]
        total_samples += r["total"]

    overall = total_correct / total_samples if total_samples > 0 else 0
    logger.okay(f"总体泛化准确率: {overall:.1%} ({total_correct}/{total_samples})")


def test_accuracy():
    """测试准确率（注意：这是用训练数据测试，可能过拟合）。"""
    logger.note("=" * 60)
    logger.note("准确率测试（训练集自测，可能过拟合）")
    logger.note("=" * 60)

    recognizer = FloorRecognizer()

    if not recognizer._load_database():
        recognizer.build_feature_database(save=True)

    logger.note(f"{recognizer}")

    locations_dir = CACHE_DIR / "locations"
    results = []

    for d in sorted(locations_dir.glob("*_minimap")):
        for floor in range(1, 4):
            if f"floor_{floor}" in d.name:
                r = recognizer.evaluate(d, floor)
                results.append(r)
                logger.mesg(
                    f"{d.name}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})"
                )
                break

    if results:
        total = sum(r["total"] for r in results)
        correct = sum(r["correct"] for r in results)
        logger.okay(f"总体: {correct/total:.1%} ({correct}/{total})")


def test_speed():
    """测试速度。"""
    logger.note("=" * 60)
    logger.note("速度测试")
    logger.note("=" * 60)

    recognizer = FloorRecognizer()
    if not recognizer._load_database():
        recognizer.build_feature_database(save=True)

    # 收集测试图像
    test_files = []
    for d in (CACHE_DIR / "locations").glob("*_minimap"):
        test_files.extend(list(d.glob("*.jpg"))[:5])

    if not test_files:
        return

    import time

    # 预热
    recognizer.recognize_file(test_files[0])

    start = time.time()
    for f in test_files:
        recognizer.recognize_file(f)
    elapsed = time.time() - start

    logger.okay(f"平均: {elapsed/len(test_files)*1000:.1f} ms/张 ({len(test_files)} 张)")


if __name__ == "__main__":
    test_speed()
    test_accuracy()  # 训练集自测（可能过拟合）
    test_cross_validation()  # 留一法交叉验证 - 测试真实泛化能力

    # python -m gtaz.recognizes
    # del gtaz\cache\floor_features.pkl 2>nul && python -m gtaz.recognizes
