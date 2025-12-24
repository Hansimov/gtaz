"""音频模板匹配模块"""

import argparse
import json
import re
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import time

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tclogger import TCLogger, logstr, dict_to_lines

# 当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR.parent / "cache"

# 模板目录
TEMPLATES_DIR = MODULE_DIR / "wavs"
# 模板文件正则表达式
TEMPLATE_REGEX = r"templatex_.*\.wav"

# 测试音频目录
SOUNDS_DIR = CACHE_DIR / "sounds"
# 测试文件开头截断时长（毫秒）
TEST_WAV_TRIM_MS = 10000
# 测试结果输出目录
TEST_JSONS_DIR = SOUNDS_DIR / "jsons"
# 测试结果绘制图像输出目录
TEST_PLOTS_DIR = SOUNDS_DIR / "plots"

# 特征X轴样本点数
FEATURE_X_POINTS = 20
# 特征Y轴样本点数
FEATURE_Y_POINTS = 512

# 特征匹配窗口（毫秒）
MATCH_WINDOW_MS = 800
# 特征匹配步长（毫秒）
MATCH_STEP_MS = 150
# 特征匹配阈值
MATCH_GATE = 0.6
# 候选者之间最小相邻时间间隔（毫秒）
CANDIDATE_MIN_OFFSET_MS = 500

# 统一采样率
UNIFIED_SAMPLE_RATE = 16000

logger = TCLogger(
    name="AudioTemplateMatcher",
    use_prefix=False,
    use_file=True,
    file_path=Path(__file__).parent / "output.log",
    file_mode="w",
)


class TemplateLoader:
    """音频模板加载"""

    def __init__(
        self,
        templates_dir: Path = TEMPLATES_DIR,
        template_regex: str = TEMPLATE_REGEX,
    ):
        self.templates_dir = templates_dir
        self.template_regex = template_regex
        self.template_files: List[Path] = []
        self.template_data: List[Tuple[int, np.ndarray]] = (
            []
        )  # [(sample_rate, data), ...]

    def scan_templates(self) -> List[Path]:
        """扫描符合正则表达式的模板文件"""
        pattern = re.compile(self.template_regex)
        self.template_files = []

        if not self.templates_dir.exists():
            logger.warn(f"模板目录不存在: {self.templates_dir}")
            return self.template_files

        for file_path in self.templates_dir.iterdir():
            if file_path.is_file() and pattern.match(file_path.name):
                self.template_files.append(file_path)

        logger.success(f"扫描到 {len(self.template_files)} 个模板文件")
        for f in self.template_files:
            logger.note(f"  - {f.name}")

        return self.template_files

    def load_templates(self) -> List[Tuple[int, np.ndarray]]:
        """加载所有模板文件"""
        self.template_data = []

        for file_path in self.template_files:
            try:
                sample_rate, data = wavfile.read(file_path)
                # 如果是立体声，转换为单声道
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                # 转换为浮点数
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128) / 128.0

                self.template_data.append((sample_rate, data))
                logger.mesg(
                    f"加载模板: {file_path.name}, "
                    f"采样率: {sample_rate}, 长度: {len(data)} samples"
                )
            except Exception as e:
                logger.warn(f"加载模板文件失败: {file_path}, 错误: {e}")

        return self.template_data


class AudioDataSamplesUnifier:
    """音频数据统一化"""

    def __init__(self, target_sample_rate: int = UNIFIED_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    def resample(
        self, data: np.ndarray, original_rate: int, target_rate: int
    ) -> np.ndarray:
        """重采样音频数据"""
        if original_rate == target_rate:
            return data

        # 计算重采样后的长度
        num_samples = int(len(data) * target_rate / original_rate)
        resampled = signal.resample(data, num_samples)
        return resampled.astype(np.float32)

    def unify(self, audio_list: List[Tuple[int, np.ndarray]]) -> List[np.ndarray]:
        """统一所有音频数据到目标采样率"""
        unified_data = []
        for sample_rate, data in audio_list:
            unified = self.resample(data, sample_rate, self.target_sample_rate)
            unified_data.append(unified)
        return unified_data

    def unify_single(self, sample_rate: int, data: np.ndarray) -> np.ndarray:
        """统一单个音频数据到目标采样率"""
        return self.resample(data, sample_rate, self.target_sample_rate)


@dataclass
class Feature:
    """音频特征数据类"""

    # 时频特征矩阵: shape = (FEATURE_X_POINTS, FEATURE_Y_POINTS)
    # X轴是时间采样点，Y轴是频率强度
    data: np.ndarray
    # 原始采样率
    sample_rate: int
    # 原始音频长度（采样点数）
    original_length: int
    # 特征对应的时间范围（毫秒）
    duration_ms: float

    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            "shape": list(self.data.shape),
            "sample_rate": self.sample_rate,
            "original_length": self.original_length,
            "duration_ms": self.duration_ms,
        }


class FeatureExtractor:
    """音频特征提取"""

    def __init__(
        self,
        x_points: int = FEATURE_X_POINTS,
        y_points: int = FEATURE_Y_POINTS,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
    ):
        self.x_points = x_points
        self.y_points = y_points
        self.sample_rate = sample_rate

    def extract(self, data: np.ndarray) -> Feature:
        """从音频数据提取时频特征"""
        # 计算STFT参数
        # nperseg: 每个段的长度，决定频率分辨率
        nperseg = self.y_points * 2
        # noverlap: 重叠长度
        noverlap = nperseg // 2

        # 计算STFT
        frequencies, times, Zxx = signal.stft(
            data,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # 取幅度谱
        magnitude = np.abs(Zxx)

        # 重采样到目标维度
        # Y轴（频率）: 从 nperseg//2+1 重采样到 y_points
        if magnitude.shape[0] != self.y_points:
            magnitude = signal.resample(magnitude, self.y_points, axis=0)

        # X轴（时间）: 重采样到 x_points
        if magnitude.shape[1] != self.x_points:
            magnitude = signal.resample(magnitude, self.x_points, axis=1)

        # 转置使得 X轴是时间，Y轴是频率
        # shape: (x_points, y_points)
        feature_data = magnitude.T

        # 归一化
        max_val = np.max(feature_data)
        if max_val > 0:
            feature_data = feature_data / max_val

        duration_ms = len(data) / self.sample_rate * 1000

        return Feature(
            data=feature_data.astype(np.float32),
            sample_rate=self.sample_rate,
            original_length=len(data),
            duration_ms=duration_ms,
        )

    def extract_window(
        self, data: np.ndarray, start_sample: int, window_samples: int
    ) -> Feature:
        """从音频数据的指定窗口提取特征"""
        end_sample = min(start_sample + window_samples, len(data))
        window_data = data[start_sample:end_sample]

        # 如果窗口数据不足，用零填充
        if len(window_data) < window_samples:
            window_data = np.pad(
                window_data, (0, window_samples - len(window_data)), mode="constant"
            )

        return self.extract(window_data)


class TemplateAligner:
    """模板对齐器 - 对多个模板进行时域对齐"""

    def __init__(self, sample_rate: int = UNIFIED_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def cross_correlate(self, ref: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
        """计算两个信号的互相关，返回最佳偏移和相关系数"""
        # 使用互相关找到最佳对齐位置
        correlation = signal.correlate(target, ref, mode="full")
        # 找到最大相关值的位置
        max_idx = np.argmax(np.abs(correlation))
        max_corr = correlation[max_idx]

        # 计算偏移量：正值表示target相对于ref需要向右移动
        offset = max_idx - (len(ref) - 1)

        # 归一化相关系数
        norm = np.sqrt(np.sum(ref**2) * np.sum(target**2))
        if norm > 0:
            corr_coef = max_corr / norm
        else:
            corr_coef = 0.0

        return offset, corr_coef

    def align_templates(
        self, templates: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """对齐所有模板到第一个模板"""
        if len(templates) == 0:
            return [], []

        if len(templates) == 1:
            return templates, [0]

        # 使用第一个模板作为参考
        reference = templates[0]
        offsets = [0]  # 第一个模板偏移为0
        aligned_templates = [reference]

        for i, template in enumerate(templates[1:], 1):
            offset, corr_coef = self.cross_correlate(reference, template)
            offsets.append(offset)
            logger.mesg(
                f"模板 {i} 对齐偏移: {offset} samples, 相关系数: {corr_coef:.4f}"
            )

            # 应用偏移
            if offset > 0:
                # target需要向右移动，即在开头填充零
                aligned = np.pad(template, (offset, 0), mode="constant")
            elif offset < 0:
                # target需要向左移动，即裁剪开头
                aligned = template[-offset:]
            else:
                aligned = template

            aligned_templates.append(aligned)

        return aligned_templates, offsets

    def find_overlap_region(
        self, templates: List[np.ndarray], offsets: List[int]
    ) -> Tuple[int, int]:
        """找到所有模板的重叠区域"""
        if len(templates) == 0:
            return 0, 0

        # 计算每个模板在统一坐标系中的起始和结束位置
        starts = []
        ends = []

        for template, offset in zip(templates, offsets):
            start = max(0, -offset) if offset < 0 else 0
            end = len(template)
            # 转换到统一坐标系
            starts.append(start + offset)
            ends.append(end + offset)

        # 重叠区域是所有模板的交集
        overlap_start = max(starts)
        overlap_end = min(ends)

        if overlap_end <= overlap_start:
            logger.warn("模板没有重叠区域")
            return 0, min(len(t) for t in templates)

        logger.mesg(
            f"重叠区域: {overlap_start} - {overlap_end}, 长度: {overlap_end - overlap_start}"
        )

        return overlap_start, overlap_end

    def extract_overlap(
        self,
        templates: List[np.ndarray],
        offsets: List[int],
        overlap_start: int,
        overlap_end: int,
    ) -> List[np.ndarray]:
        """从对齐后的模板中提取重叠区域"""
        extracted = []

        for template, offset in zip(templates, offsets):
            # 计算在原始模板中的对应位置
            local_start = overlap_start - offset
            local_end = overlap_end - offset

            # 确保索引在有效范围内
            local_start = max(0, local_start)
            local_end = min(len(template), local_end)

            extracted.append(template[local_start:local_end])

        return extracted


class TemplateFuser:
    """模板融合器 - 将多个对齐后的模板特征融合为一个"""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
    ):
        self.feature_extractor = feature_extractor

    def fuse_templates(
        self, templates: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> Feature:
        """融合多个对齐后的模板"""
        if len(templates) == 0:
            raise ValueError("没有模板可融合")

        if len(templates) == 1:
            return self.feature_extractor.extract(templates[0])

        # 默认使用均等权重
        if weights is None:
            weights = [1.0 / len(templates)] * len(templates)

        # 确保所有模板长度相同
        min_len = min(len(t) for t in templates)
        templates = [t[:min_len] for t in templates]

        # 加权融合音频数据
        fused_audio = np.zeros(min_len, dtype=np.float32)
        for template, weight in zip(templates, weights):
            fused_audio += template * weight

        # 提取融合后的特征
        return self.feature_extractor.extract(fused_audio)

    def fuse_features(
        self, features: List[Feature], weights: Optional[List[float]] = None
    ) -> Feature:
        """直接融合特征矩阵"""
        if len(features) == 0:
            raise ValueError("没有特征可融合")

        if len(features) == 1:
            return features[0]

        # 默认使用均等权重
        if weights is None:
            weights = [1.0 / len(features)] * len(features)

        # 加权融合特征矩阵
        fused_data = np.zeros_like(features[0].data, dtype=np.float32)
        for feature, weight in zip(features, weights):
            fused_data += feature.data * weight

        # 归一化
        max_val = np.max(fused_data)
        if max_val > 0:
            fused_data = fused_data / max_val

        return Feature(
            data=fused_data,
            sample_rate=features[0].sample_rate,
            original_length=int(np.mean([f.original_length for f in features])),
            duration_ms=np.mean([f.duration_ms for f in features]),
        )


@dataclass
class MatchCandidate:
    """音频模板匹配候选数据类"""

    # 匹配起始位置（毫秒）
    start_ms: float
    # 匹配结束位置（毫秒）
    end_ms: float
    # 匹配分数 (0-1)
    score: float
    # 起始采样点
    start_sample: int
    # 结束采样点
    end_sample: int

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "start_ms": round(self.start_ms, 2),
            "end_ms": round(self.end_ms, 2),
            "score": round(self.score, 4),
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }


@dataclass
class MatchResult:
    """音频模板匹配结果数据类"""

    # 测试文件路径
    test_file: str
    # 测试音频总时长（毫秒）
    total_duration_ms: float
    # 采样率
    sample_rate: int
    # 匹配候选列表
    candidates: List[MatchCandidate] = field(default_factory=list)
    # 匹配参数
    match_window_ms: float = MATCH_WINDOW_MS
    match_step_ms: float = MATCH_STEP_MS
    match_gate: float = MATCH_GATE
    candidate_min_offset_ms: float = CANDIDATE_MIN_OFFSET_MS

    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            "test_file": self.test_file,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "sample_rate": self.sample_rate,
            "candidates": [c.to_dict() for c in self.candidates],
            "params": {
                "match_window_ms": self.match_window_ms,
                "match_step_ms": self.match_step_ms,
                "match_gate": self.match_gate,
                "candidate_min_offset_ms": self.candidate_min_offset_ms,
            },
        }


class FeatureMatcher:
    """音频特征匹配"""

    def __init__(
        self,
        template_feature: Feature,
        window_ms: float = MATCH_WINDOW_MS,
        step_ms: float = MATCH_STEP_MS,
        gate: float = MATCH_GATE,
        min_offset_ms: float = CANDIDATE_MIN_OFFSET_MS,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
    ):
        self.template_feature = template_feature
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.gate = gate
        self.min_offset_ms = min_offset_ms
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)

    def compute_similarity(self, feature1: Feature, feature2: Feature) -> float:
        """计算两个特征的相似度（归一化互相关）"""
        f1 = feature1.data.flatten()
        f2 = feature2.data.flatten()

        # 零均值归一化
        f1 = f1 - np.mean(f1)
        f2 = f2 - np.mean(f2)

        # 计算归一化互相关
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(f1, f2) / (norm1 * norm2)
        # 将相似度映射到 [0, 1] 范围
        similarity = (similarity + 1) / 2

        return float(similarity)

    def match(self, test_data: np.ndarray) -> MatchResult:
        """在测试数据中匹配模板"""
        total_duration_ms = len(test_data) / self.sample_rate * 1000

        # 计算滑动窗口参数
        window_samples = int(self.window_ms * self.sample_rate / 1000)
        step_samples = int(self.step_ms * self.sample_rate / 1000)

        # 存储所有候选
        all_candidates: List[MatchCandidate] = []

        # 滑动窗口匹配
        start_sample = 0
        while start_sample + window_samples <= len(test_data):
            # 提取窗口特征
            window_feature = self.feature_extractor.extract_window(
                test_data, start_sample, window_samples
            )

            # 计算相似度
            score = self.compute_similarity(self.template_feature, window_feature)

            # 如果超过阈值，添加候选
            if score >= self.gate:
                start_ms = start_sample / self.sample_rate * 1000
                end_ms = (start_sample + window_samples) / self.sample_rate * 1000
                candidate = MatchCandidate(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=score,
                    start_sample=start_sample,
                    end_sample=start_sample + window_samples,
                )
                all_candidates.append(candidate)

            start_sample += step_samples

        # 去除重叠的候选（非极大值抑制）
        filtered_candidates = self._non_max_suppression(all_candidates)

        return MatchResult(
            test_file="",
            total_duration_ms=total_duration_ms,
            sample_rate=self.sample_rate,
            candidates=filtered_candidates,
            match_window_ms=self.window_ms,
            match_step_ms=self.step_ms,
            match_gate=self.gate,
            candidate_min_offset_ms=self.min_offset_ms,
        )

    def _non_max_suppression(
        self, candidates: List[MatchCandidate]
    ) -> List[MatchCandidate]:
        """非极大值抑制，去除重叠的候选"""
        if len(candidates) == 0:
            return []

        # 按分数降序排序
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected = []

        for candidate in sorted_candidates:
            # 检查是否与已选择的候选重叠
            is_overlapping = False
            for selected_candidate in selected:
                # 计算时间间隔
                time_diff = abs(candidate.start_ms - selected_candidate.start_ms)
                if time_diff < self.min_offset_ms:
                    is_overlapping = True
                    break

            if not is_overlapping:
                selected.append(candidate)

        # 按时间排序
        selected.sort(key=lambda c: c.start_ms)

        return selected


class TestDataSamplesLoader:
    """测试数据加载和预处理"""

    def __init__(
        self,
        sounds_dir: Path = SOUNDS_DIR,
        trim_ms: int = TEST_WAV_TRIM_MS,
        target_sample_rate: int = UNIFIED_SAMPLE_RATE,
    ):
        self.sounds_dir = sounds_dir
        self.trim_ms = trim_ms
        self.target_sample_rate = target_sample_rate
        self.unifier = AudioDataSamplesUnifier(target_sample_rate)

    def scan_test_files(self) -> List[Path]:
        """扫描测试WAV文件"""
        test_files = []

        if not self.sounds_dir.exists():
            logger.warn(f"测试目录不存在: {self.sounds_dir}")
            return test_files

        # 递归查找所有WAV文件
        for wav_file in self.sounds_dir.rglob("*.wav"):
            test_files.append(wav_file)

        logger.success(f"扫描到 {len(test_files)} 个测试文件")
        return test_files

    def load_test_file(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """加载单个测试文件"""
        try:
            sample_rate, data = wavfile.read(file_path)

            # 如果是立体声，转换为单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 转换为浮点数
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0

            # 截断开头
            if self.trim_ms > 0:
                trim_samples = int(self.trim_ms * sample_rate / 1000)
                if trim_samples < len(data):
                    data = data[trim_samples:]

            # 统一采样率
            data = self.unifier.unify_single(sample_rate, data)

            logger.mesg(
                f"加载测试文件: {file_path.name}, "
                f"长度: {len(data) / self.target_sample_rate:.2f}s"
            )

            return data, self.target_sample_rate

        except Exception as e:
            logger.warn(f"加载测试文件失败: {file_path}, 错误: {e}")
            return np.array([]), self.target_sample_rate


class MatchResultsPlotter:
    """匹配结果绘制"""

    def __init__(
        self,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate

    def plot(
        self,
        test_data: np.ndarray,
        match_result: MatchResult,
        output_path: Path,
        title: str = "Audio Template Matching Result",
    ):
        """绘制匹配结果"""
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 6))

        # 计算时间轴（毫秒）
        time_ms = np.arange(len(test_data)) / self.sample_rate * 1000

        # 绘制波形
        ax.plot(time_ms, test_data, "b-", linewidth=0.5, alpha=0.7, label="Waveform")

        # 绘制匹配区域
        y_min, y_max = ax.get_ylim()
        rect_height = y_max - y_min

        for i, candidate in enumerate(match_result.candidates):
            # 绘制矩形框
            rect = plt.Rectangle(
                (candidate.start_ms, y_min),
                candidate.end_ms - candidate.start_ms,
                rect_height,
                fill=True,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.3,
                linewidth=2,
            )
            ax.add_patch(rect)

            # 添加分数文本
            text_x = (candidate.start_ms + candidate.end_ms) / 2
            text_y = y_max * 0.8
            ax.text(
                text_x,
                text_y,
                f"{candidate.score:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="darkred",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # 设置标签和标题
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # 添加匹配信息
        info_text = f"Total matches: {len(match_result.candidates)}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.success(f"保存匹配结果图像: {output_path}")


class FeatureMatchTester:
    """音频特征匹配测试"""

    def __init__(
        self,
        templates_dir: Path = TEMPLATES_DIR,
        template_regex: str = TEMPLATE_REGEX,
        sounds_dir: Path = SOUNDS_DIR,
        jsons_dir: Path = TEST_JSONS_DIR,
        plots_dir: Path = TEST_PLOTS_DIR,
        trim_ms: int = TEST_WAV_TRIM_MS,
        window_ms: float = MATCH_WINDOW_MS,
        step_ms: float = MATCH_STEP_MS,
        gate: float = MATCH_GATE,
        min_offset_ms: float = CANDIDATE_MIN_OFFSET_MS,
    ):
        self.templates_dir = templates_dir
        self.template_regex = template_regex
        self.sounds_dir = sounds_dir
        self.jsons_dir = jsons_dir
        self.plots_dir = plots_dir
        self.trim_ms = trim_ms
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.gate = gate
        self.min_offset_ms = min_offset_ms

        # 初始化组件
        self.template_loader = TemplateLoader(templates_dir, template_regex)
        self.unifier = AudioDataSamplesUnifier()
        self.feature_extractor = FeatureExtractor()
        self.aligner = TemplateAligner()
        self.fuser = TemplateFuser(self.feature_extractor)
        self.test_loader = TestDataSamplesLoader(sounds_dir, trim_ms)
        self.plotter = MatchResultsPlotter()

        # 融合后的模板特征
        self.fused_template_feature: Optional[Feature] = None

    def load_and_fuse_templates(self) -> Feature:
        """加载并融合模板"""
        logger.hint("=" * 50)
        logger.hint("加载模板文件...")

        # 扫描并加载模板
        self.template_loader.scan_templates()
        template_data = self.template_loader.load_templates()

        if len(template_data) == 0:
            raise ValueError("没有找到模板文件")

        # 统一采样率
        logger.hint("统一音频采样率...")
        unified_templates = self.unifier.unify(template_data)

        # 对齐模板
        logger.hint("对齐模板...")
        aligned_templates, offsets = self.aligner.align_templates(unified_templates)

        # 找到重叠区域
        overlap_start, overlap_end = self.aligner.find_overlap_region(
            aligned_templates, offsets
        )

        # 提取重叠区域
        extracted_templates = self.aligner.extract_overlap(
            unified_templates, offsets, overlap_start, overlap_end
        )

        # 融合模板
        logger.hint("融合模板特征...")
        self.fused_template_feature = self.fuser.fuse_templates(extracted_templates)

        logger.success(
            f"模板特征融合完成, 特征维度: {self.fused_template_feature.data.shape}"
        )

        return self.fused_template_feature

    def get_output_filename(self, test_file: Path) -> str:
        """生成输出文件名"""
        parent_name = test_file.parent.name
        file_stem = test_file.stem
        return f"{parent_name}_{file_stem}"

    def test_single_file(self, test_file: Path) -> MatchResult:
        """测试单个文件"""
        logger.hint("-" * 40)
        logger.hint(f"测试文件: {test_file}")

        # 加载测试数据
        test_data, sample_rate = self.test_loader.load_test_file(test_file)

        if len(test_data) == 0:
            logger.warn(f"测试文件数据为空: {test_file}")
            return MatchResult(
                test_file=str(test_file),
                total_duration_ms=0,
                sample_rate=sample_rate,
            )

        # 创建匹配器
        matcher = FeatureMatcher(
            template_feature=self.fused_template_feature,
            window_ms=self.window_ms,
            step_ms=self.step_ms,
            gate=self.gate,
            min_offset_ms=self.min_offset_ms,
        )

        # 执行匹配
        result = matcher.match(test_data)
        result.test_file = str(test_file)

        logger.success(f"匹配完成, 找到 {len(result.candidates)} 个候选")

        # 生成输出文件名
        output_name = self.get_output_filename(test_file)

        # 保存JSON结果
        json_path = self.jsons_dir / f"{output_name}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.success(f"保存JSON结果: {json_path}")

        # 绘制并保存图像
        plot_path = self.plots_dir / f"{output_name}.jpg"
        self.plotter.plot(
            test_data,
            result,
            plot_path,
            title=f"Match Result: {test_file.name}",
        )

        return result

    def test_all_files(self) -> List[MatchResult]:
        """测试所有文件"""
        # 首先加载并融合模板
        self.load_and_fuse_templates()

        # 扫描测试文件
        test_files = self.test_loader.scan_test_files()

        if len(test_files) == 0:
            logger.warn("没有找到测试文件")
            return []

        # 测试所有文件
        results = []
        for test_file in test_files:
            result = self.test_single_file(test_file)
            results.append(result)

        logger.hint("=" * 50)
        logger.success(f"测试完成, 共处理 {len(results)} 个文件")

        return results


class TemplateMatcherArgParser:
    """命令行参数解析"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="音频模板匹配工具",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._setup_arguments()

    def _setup_arguments(self):
        """设置命令行参数"""
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="运行测试模式",
        )
        self.parser.add_argument(
            "--templates-dir",
            type=str,
            default=str(TEMPLATES_DIR),
            help="模板文件目录",
        )
        self.parser.add_argument(
            "--template-regex",
            type=str,
            default=TEMPLATE_REGEX,
            help="模板文件正则表达式",
        )
        self.parser.add_argument(
            "--sounds-dir",
            type=str,
            default=str(SOUNDS_DIR),
            help="测试音频目录",
        )
        self.parser.add_argument(
            "--trim-ms",
            type=int,
            default=TEST_WAV_TRIM_MS,
            help="测试文件开头截断时长（毫秒）",
        )
        self.parser.add_argument(
            "--window-ms",
            type=float,
            default=MATCH_WINDOW_MS,
            help="匹配窗口大小（毫秒）",
        )
        self.parser.add_argument(
            "--step-ms",
            type=float,
            default=MATCH_STEP_MS,
            help="匹配步长（毫秒）",
        )
        self.parser.add_argument(
            "--gate",
            type=float,
            default=MATCH_GATE,
            help="匹配阈值",
        )
        self.parser.add_argument(
            "--min-offset-ms",
            type=float,
            default=CANDIDATE_MIN_OFFSET_MS,
            help="候选者最小间隔（毫秒）",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数"""
        return self.parser.parse_args()


def main():
    """主函数"""
    arg_parser = TemplateMatcherArgParser()
    args = arg_parser.parse()

    if args.test:
        logger.success("启动音频模板匹配测试...")

        tester = FeatureMatchTester(
            templates_dir=Path(args.templates_dir),
            template_regex=args.template_regex,
            sounds_dir=Path(args.sounds_dir),
            trim_ms=args.trim_ms,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            gate=args.gate,
            min_offset_ms=args.min_offset_ms,
        )

        results = tester.test_all_files()

        # 输出摘要
        total_candidates = sum(len(r.candidates) for r in results)
        logger.hint("=" * 50)
        logger.success(f"测试摘要:")
        logger.mesg(f"  - 测试文件数: {len(results)}")
        logger.mesg(f"  - 总匹配数: {total_candidates}")
    else:
        arg_parser.parser.print_help()


if __name__ == "__main__":
    main()

    # python -m gtaz.audios.signals_v3 --test
