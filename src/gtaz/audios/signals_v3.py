"""音频模板匹配模块"""

import argparse
import json
import re
import time
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tclogger import TCLogger

# 当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR.parent / "cache"

# 模板目录
TEMPLATES_DIR = CACHE_DIR / "wavs"
# 模板文件正则表达式
TEMPLATE_REGEX = r"templatey_.*\.wav"

# 测试音频目录
SOUNDS_DIR = CACHE_DIR / "sounds"
# 测试文件开头截断时长（毫秒）
TEST_WAV_TRIM_MS = 10000
# 测试结果输出目录
TEST_JSONS_DIR = SOUNDS_DIR / "jsons"
# 测试结果绘制图像输出目录
TEST_PLOTS_DIR = SOUNDS_DIR / "plots"

# 特征X轴样本点数（时间分辨率，越高越能捕捉尖峰）
FEATURE_X_POINTS = 64
# 特征Y轴样本点数（频率分辨率）
FEATURE_Y_POINTS = 64

# 特征匹配窗口（毫秒）- 根据模板时长自动调整
MATCH_WINDOW_MS = 800
# 特征匹配步长（毫秒）
MATCH_STEP_MS = 100
# 模板融合后的最小时长（毫秒）
MIN_TEMPLATE_DURATION_MS = 600
# 特征匹配阈值（相关系数，范围[0,1]）
MATCH_GATE = 0.42
# 候选者之间最小相邻时间间隔（毫秒）
CANDIDATE_MIN_OFFSET_MS = int(MATCH_WINDOW_MS / 2)
# 音量比例阈值，测试信号音量低于模板的此比例则不匹配
# 例如 0.5 表示测试信号音量低于模板的50%时过滤
VOLUME_RATIO_THRESHOLD = 0.45

# 统一采样率（44100Hz可支持最高22050Hz的频率）
UNIFIED_SAMPLE_RATE = 44100
# 低通滤波截止频率
LOW_PASS_CUTOFF_HZ = 1000
# 高通滤波截止频率
HIGH_PASS_CUTOFF_HZ = 400

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

        logger.mesg(f"扫描到 {len(self.template_files)} 个模板文件")

        return self.template_files

    def load_templates(self) -> List[Tuple[int, np.ndarray]]:
        """加载所有模板文件"""
        self.template_data = []

        for file_path in self.template_files:
            try:
                sample_rate, data = wavfile.read(file_path)
                # 先转换为浮点数（归一化），再转单声道
                # 注意：必须先归一化再 np.mean，否则 np.mean 会把 int16 转成 float64 但值不变
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128) / 128.0
                else:
                    data = data.astype(np.float32)

                # 如果是立体声，转换为单声道
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1).astype(np.float32)

                self.template_data.append((sample_rate, data))
            except Exception as e:
                logger.warn(f"加载模板文件失败: {file_path}, 错误: {e}")

        return self.template_data


class AudioDataSamplesUnifier:
    """音频数据统一化，包含重采样和滤波"""

    def __init__(
        self,
        target_sample_rate: int = UNIFIED_SAMPLE_RATE,
        filter_type: str = "none",  # 滤波器类型: "lowpass", "highpass", "none"
        filter_freq: float = None,  # 滤波截止频率
    ):
        self.target_sample_rate = target_sample_rate
        self.filter_type = filter_type.lower()
        # 默认截止频率
        if filter_freq is None:
            if self.filter_type == "lowpass":
                self.filter_freq = LOW_PASS_CUTOFF_HZ
            elif self.filter_type == "highpass":
                self.filter_freq = HIGH_PASS_CUTOFF_HZ
            else:
                self.filter_freq = 0
        else:
            self.filter_freq = filter_freq

    def apply_filter(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """应用滤波器"""
        if self.filter_type == "none" or self.filter_freq <= 0:
            return data

        # 计算归一化截止频率（相对于奈奎斯特频率）
        nyquist = sample_rate / 2
        if self.filter_freq >= nyquist:
            # 截止频率超过奈奎斯特频率，无法滤波
            return data

        normalized_freq = self.filter_freq / nyquist

        # 设计巴特沃斯滤波器（4阶）
        if self.filter_type == "lowpass":
            b, a = signal.butter(4, normalized_freq, btype="low")
        elif self.filter_type == "highpass":
            b, a = signal.butter(4, normalized_freq, btype="high")
        else:
            return data

        # 应用滤波器
        filtered = signal.filtfilt(b, a, data)
        return filtered.astype(np.float32)

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
        """统一所有音频数据到目标采样率，并应用滤波"""
        unified_data = []
        for sample_rate, data in audio_list:
            # 先重采样
            unified = self.resample(data, sample_rate, self.target_sample_rate)
            # 再应用滤波
            unified = self.apply_filter(unified, self.target_sample_rate)
            unified_data.append(unified)
        return unified_data

    def unify_single(self, sample_rate: int, data: np.ndarray) -> np.ndarray:
        """统一单个音频数据到目标采样率，并应用滤波"""
        # 先重采样
        unified = self.resample(data, sample_rate, self.target_sample_rate)
        # 再应用滤波
        unified = self.apply_filter(unified, self.target_sample_rate)
        return unified


@dataclass
class Feature:
    """音频特征数据类

    包含三类特征：
    1. 时频特征矩阵 (data): 频谱图，shape = (FEATURE_X_POINTS, FEATURE_Y_POINTS)
    2. 时域包络 (temporal_envelope): 每个时间帧的总能量，用于检测尖峰
    3. 频谱带宽 (spectral_bandwidth): 每个时间帧的频谱扩散度，全频谱时值大，低频时值小

    额外的辅助特征：
    - energy: 频谱能量 (RMS)
    - mean_magnitude: 对数幅度均值
    - peak_prominence: 尖峰突出度（峰值与均值之比）
    - volume: 原始时域信号的 RMS 音量，用于音量过滤
    """

    # 时频特征矩阵: shape = (FEATURE_X_POINTS, FEATURE_Y_POINTS)
    # X轴是时间采样点，Y轴是频率强度
    data: np.ndarray
    # 时域包络: shape = (FEATURE_X_POINTS,)
    # 每个时间帧的总能量，用于检测尖峰
    temporal_envelope: np.ndarray
    # 频谱带宽: shape = (FEATURE_X_POINTS,)
    # 每个时间帧的频谱扩散度（标准差），全频谱信号时值大
    spectral_bandwidth: np.ndarray
    # 原始采样率
    sample_rate: int
    # 原始音频长度（采样点数）
    original_length: int
    # 特征对应的时间范围（毫秒）
    duration_ms: float
    # 特征能量（频谱RMS值），用于判断信号强度
    energy: float = 0.0
    # 特征幅度均值（log magnitude的均值）
    mean_magnitude: float = 0.0
    # 尖峰突出度（时域包络的峰值与均值之比）
    peak_prominence: float = 0.0
    # 音量（原始时域信号的RMS值），用于绝对音量过滤
    volume: float = 0.0

    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            "shape": list(self.data.shape),
            "sample_rate": self.sample_rate,
            "original_length": self.original_length,
            "duration_ms": self.duration_ms,
            "energy": round(self.energy, 6),
            "mean_magnitude": round(self.mean_magnitude, 4),
            "peak_prominence": round(self.peak_prominence, 4),
            "volume": round(self.volume, 6),
        }


class FeatureExtractor:
    """音频特征提取

    使用线性Hz频率刻度，高频权重更高。
    直接使用STFT频谱，不使用Mel变换。
    """

    def __init__(
        self,
        x_points: int = FEATURE_X_POINTS,
        y_points: int = FEATURE_Y_POINTS,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
        n_fft: int = 2048,
        f_min: float = 20.0,
        f_max: float = None,  # None表示使用奈奎斯特频率(sample_rate/2)
        high_freq_weight: float = 2.0,  # 高频权重倍数
    ):
        self.x_points = x_points
        self.y_points = y_points
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        # 如果f_max为None，使用奈奎斯特频率
        self.f_max = f_max if f_max is not None else sample_rate / 2
        self.high_freq_weight = high_freq_weight

        # 预计算频率权重（线性增加，高频权重更高）
        self.freq_weights = self._create_freq_weights()

    def _create_freq_weights(self) -> np.ndarray:
        """创建频率权重，高频权重更高"""
        # 权重从1线性增加到high_freq_weight
        weights = np.linspace(1.0, self.high_freq_weight, self.y_points)
        return weights

    def extract(self, data: np.ndarray) -> Feature:
        """从音频数据提取频谱特征

        提取三类特征：
        1. 时频特征矩阵：频谱图 (x_points, y_points)
        2. 时域包络：每个时间帧的总能量 (x_points,)
        3. 频谱带宽：每个时间帧的频谱扩散度 (x_points,)

        额外计算：
        - volume: 原始时域信号的 RMS 音量，用于后续的音量过滤
        """
        # === 计算原始时域信号的音量（RMS）===
        volume = float(np.sqrt(np.mean(data**2)))

        # 如果数据太短，进行零填充以满足n_fft要求
        min_length = self.n_fft
        if len(data) < min_length:
            data = np.pad(data, (0, min_length - len(data)), mode="constant")

        # 计算STFT参数
        hop_length = self.n_fft // 4

        # 计算STFT
        frequencies, times, Zxx = signal.stft(
            data,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - hop_length,
        )

        # 取幅度谱
        magnitude = np.abs(Zxx)

        # 限制频率范围
        freq_bin_min = int(self.f_min * self.n_fft / self.sample_rate)
        freq_bin_max = int(self.f_max * self.n_fft / self.sample_rate)
        freq_bin_max = min(freq_bin_max, magnitude.shape[0])

        # 截取感兴趣的频率范围
        magnitude = magnitude[freq_bin_min:freq_bin_max, :]
        freq_range = frequencies[freq_bin_min:freq_bin_max]

        # === 计算时域包络（在重采样前计算，保留原始分辨率的信息）===
        # 每个时间帧的总能量（沿频率轴求和）
        frame_energy = np.sum(magnitude, axis=0)

        # === 计算频谱带宽（频谱质心的标准差）- 向量化版本 ===
        # 频谱质心：能量加权的频率中心
        # 频谱带宽：能量加权的频率标准差
        total_energy = np.sum(magnitude, axis=0)  # shape: (time_frames,)
        # 避免除零
        total_energy_safe = np.where(total_energy > 1e-10, total_energy, 1.0)
        # 归一化为概率分布
        prob = magnitude / total_energy_safe  # shape: (freq_bins, time_frames)
        # 频谱质心 - 向量化计算
        centroid = np.sum(
            freq_range[:, np.newaxis] * prob, axis=0
        )  # shape: (time_frames,)
        # 频谱带宽（标准差）- 向量化计算
        variance = np.sum(((freq_range[:, np.newaxis] - centroid) ** 2) * prob, axis=0)
        spectral_bandwidth_raw = np.sqrt(variance)
        # 将能量过低的帧的带宽设为0
        spectral_bandwidth_raw = np.where(
            total_energy > 1e-10, spectral_bandwidth_raw, 0.0
        )

        # Y轴（频率）: 重采样到 y_points
        if magnitude.shape[0] != self.y_points:
            magnitude = signal.resample(magnitude, self.y_points, axis=0)

        # X轴（时间）: 重采样到 x_points
        if magnitude.shape[1] != self.x_points:
            magnitude = signal.resample(magnitude, self.x_points, axis=1)
            frame_energy = signal.resample(frame_energy, self.x_points)
            spectral_bandwidth_raw = signal.resample(
                spectral_bandwidth_raw, self.x_points
            )

        # 确保非负
        frame_energy = np.maximum(frame_energy, 0)
        spectral_bandwidth_raw = np.maximum(spectral_bandwidth_raw, 0)

        # 对数压缩（使用np.maximum确保最小值，避免log10的警告）
        magnitude = np.maximum(magnitude, 1e-10)
        log_magnitude = np.log10(magnitude)

        # 计算能量信息
        energy = float(np.sqrt(np.mean(magnitude**2)))
        mean_magnitude = float(np.mean(log_magnitude))

        # === 计算时域包络特征 ===
        # 归一化时域包络（相对于均值）
        envelope_mean = np.mean(frame_energy)
        if envelope_mean > 1e-10:
            temporal_envelope = frame_energy / envelope_mean
        else:
            temporal_envelope = np.ones(self.x_points)

        # === 计算尖峰突出度 ===
        # 峰值与均值之比，值越大说明有明显尖峰
        peak_prominence = float(np.max(temporal_envelope))

        # === 归一化频谱带宽 ===
        # 归一化到 [0, 1] 范围，相对于最大可能带宽
        max_bandwidth = (self.f_max - self.f_min) / 2
        if max_bandwidth > 0:
            spectral_bandwidth = spectral_bandwidth_raw / max_bandwidth
        else:
            spectral_bandwidth = np.zeros(self.x_points)

        # 应用高频权重（沿频率轴）
        weighted_magnitude = log_magnitude * self.freq_weights[:, np.newaxis]

        # 转置使得 X轴是时间，Y轴是频率
        # shape: (x_points, y_points)
        feature_data = weighted_magnitude.T

        duration_ms = len(data) / self.sample_rate * 1000

        return Feature(
            data=feature_data.astype(np.float32),
            temporal_envelope=temporal_envelope.astype(np.float32),
            spectral_bandwidth=spectral_bandwidth.astype(np.float32),
            sample_rate=self.sample_rate,
            original_length=len(data),
            duration_ms=duration_ms,
            energy=energy,
            mean_magnitude=mean_magnitude,
            peak_prominence=peak_prominence,
            volume=volume,
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
    """模板对齐器 - 对多个模板进行时域对齐

    使用参考模板（最长模板）的中心区域进行对齐，而非严格的全体交集，
    以保证融合后的模板有足够的时长（至少 MIN_TEMPLATE_DURATION_MS）。
    """

    def __init__(
        self,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
        min_duration_ms: float = MIN_TEMPLATE_DURATION_MS,
    ):
        self.sample_rate = sample_rate
        self.min_duration_ms = min_duration_ms
        self.min_duration_samples = int(min_duration_ms * sample_rate / 1000)

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
        """对齐所有模板到最长模板的中心区域

        返回: (对齐后的模板列表, 偏移量列表)
        偏移量是相对于参考模板的采样点偏移
        """
        if len(templates) == 0:
            return [], []

        if len(templates) == 1:
            return templates, [0]

        # 选择最长的模板作为参考
        ref_idx = max(range(len(templates)), key=lambda i: len(templates[i]))
        reference = templates[ref_idx]

        offsets = []
        for i, template in enumerate(templates):
            if i == ref_idx:
                offsets.append(0)
            else:
                offset, corr_coef = self.cross_correlate(reference, template)
                offsets.append(offset)

        return templates, offsets

    def find_center_region(
        self, templates: List[np.ndarray], offsets: List[int]
    ) -> Tuple[int, int]:
        """找到基于参考模板中心的融合区域

        策略：
        1. 选择最长模板的中心区域作为基准
        2. 区域长度为 max(min_duration_samples, 最小模板长度)
        3. 确保区域在参考模板范围内
        """
        if len(templates) == 0:
            return 0, 0

        # 找到最长模板的索引
        ref_idx = max(range(len(templates)), key=lambda i: len(templates[i]))
        ref_length = len(templates[ref_idx])

        # 计算目标区域长度
        min_template_len = min(len(t) for t in templates)
        target_length = max(self.min_duration_samples, min_template_len)
        target_length = min(target_length, ref_length)  # 不能超过参考模板长度

        # 计算中心区域的起始和结束位置（相对于参考模板）
        center = ref_length // 2
        region_start = max(0, center - target_length // 2)
        region_end = min(ref_length, region_start + target_length)

        # 如果结束位置受限，重新调整起始位置
        if region_end - region_start < target_length:
            region_start = max(0, region_end - target_length)

        duration_ms = (region_end - region_start) / self.sample_rate * 1000
        logger.mesg(
            f"融合区域: {region_start} - {region_end}, "
            f"长度: {region_end - region_start} samples ({duration_ms:.1f}ms)"
        )

        return region_start, region_end

    def extract_aligned_region(
        self,
        templates: List[np.ndarray],
        offsets: List[int],
        region_start: int,
        region_end: int,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """从各模板中提取对齐后的区域

        返回: (提取的模板列表, 每个模板的覆盖率权重)
        对于没有完全覆盖区域的模板，缺失部分用零填充，权重降低
        """
        extracted = []
        weights = []
        region_length = region_end - region_start

        for template, offset in zip(templates, offsets):
            # 计算在当前模板中的对应位置
            # offset > 0 表示当前模板相对于参考模板向右偏移
            # 所以参考模板的 region_start 对应当前模板的 region_start - offset
            local_start = region_start - offset
            local_end = region_end - offset

            # 创建输出数组（初始化为零）
            output = np.zeros(region_length, dtype=np.float32)

            # 计算有效的提取范围
            valid_start = max(0, local_start)
            valid_end = min(len(template), local_end)

            if valid_end > valid_start:
                # 计算在输出数组中的对应位置
                out_start = valid_start - local_start
                out_end = out_start + (valid_end - valid_start)

                # 复制有效数据
                output[out_start:out_end] = template[valid_start:valid_end]

                # 计算覆盖率作为权重
                coverage = (valid_end - valid_start) / region_length
            else:
                coverage = 0.0

            extracted.append(output)
            weights.append(coverage)

        return extracted, weights


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
        """融合多个对齐后的模板

        Args:
            templates: 对齐后的模板列表（长度相同）
            weights: 每个模板的权重（覆盖率），用于加权平均
        """
        if len(templates) == 0:
            raise ValueError("没有模板可融合")

        if len(templates) == 1:
            return self.feature_extractor.extract(templates[0])

        # 默认使用均等权重
        if weights is None:
            weights = [1.0] * len(templates)

        # 确保所有模板长度相同
        min_len = min(len(t) for t in templates)
        templates = [t[:min_len] for t in templates]

        # 加权融合音频数据（按覆盖率加权平均）
        fused_audio = np.zeros(min_len, dtype=np.float32)
        total_weight = sum(weights)

        if total_weight > 0:
            for template, weight in zip(templates, weights):
                fused_audio += template * weight
            fused_audio /= total_weight

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

        # 融合时域包络
        fused_temporal_envelope = np.zeros_like(
            features[0].temporal_envelope, dtype=np.float32
        )
        for feature, weight in zip(features, weights):
            fused_temporal_envelope += feature.temporal_envelope * weight

        # 融合频谱带宽
        fused_spectral_bandwidth = np.zeros_like(
            features[0].spectral_bandwidth, dtype=np.float32
        )
        for feature, weight in zip(features, weights):
            fused_spectral_bandwidth += feature.spectral_bandwidth * weight

        # 融合能量信息（加权平均）
        fused_energy = sum(f.energy * w for f, w in zip(features, weights))
        fused_mean_magnitude = sum(
            f.mean_magnitude * w for f, w in zip(features, weights)
        )
        fused_peak_prominence = sum(
            f.peak_prominence * w for f, w in zip(features, weights)
        )
        fused_volume = sum(f.volume * w for f, w in zip(features, weights))

        return Feature(
            data=fused_data,
            temporal_envelope=fused_temporal_envelope,
            spectral_bandwidth=fused_spectral_bandwidth,
            sample_rate=features[0].sample_rate,
            original_length=int(np.mean([f.original_length for f in features])),
            duration_ms=np.mean([f.duration_ms for f in features]),
            energy=fused_energy,
            mean_magnitude=fused_mean_magnitude,
            peak_prominence=fused_peak_prominence,
            volume=fused_volume,
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
        """计算两个特征的相似度（综合时域、频域和带宽特征）

        返回值范围: [0, 1]
        - 1: 完全匹配
        - 0: 不匹配

        算法（针对"全频谱尖峰 + 两侧低频信号"模式优化）:
        1. 音量过滤: 测试信号音量不能比模板低于 VOLUME_RATIO_THRESHOLD 倍（使用原始RMS音量）
        2. 时域包络相似度: 检测是否有相似的尖峰模式（权重最高）
        3. 频谱带宽相似度: 检测尖峰处是否为全频谱信号
        4. 频谱形状相似度: 整体频谱形状匹配
        5. 尖峰突出度匹配: 确保测试信号也有明显尖峰
        """
        # === 0. 音量过滤（使用原始时域RMS音量）===
        # feature1是模板，feature2是测试信号
        template_volume = feature1.volume
        test_volume = feature2.volume

        # 计算音量比例
        if template_volume > 1e-10:
            volume_ratio = test_volume / template_volume
        else:
            volume_ratio = 1.0

        # 如果测试信号音量低于模板的阈值比例，直接返回0
        if volume_ratio < VOLUME_RATIO_THRESHOLD:
            return 0.0

        # 音量惩罚因子：测试信号音量稍低时的软惩罚
        # 从阈值到1之间线性惩罚
        if volume_ratio < 1.0:
            # 线性映射：阈值->0, 1.0->1.0
            volume_penalty = (volume_ratio - VOLUME_RATIO_THRESHOLD) / (
                1.0 - VOLUME_RATIO_THRESHOLD
            )
        else:
            # 测试信号音量更高，不惩罚
            volume_penalty = 1.0

        # === 1. 时域包络相似度（权重最高）===
        # 检测时域包络的形状是否匹配（尖峰位置和形状）
        env1 = feature1.temporal_envelope
        env2 = feature2.temporal_envelope

        # 归一化到相同尺度
        env1_norm = env1 / (np.max(env1) + 1e-10)
        env2_norm = env2 / (np.max(env2) + 1e-10)

        # 皮尔逊相关系数
        env1_centered = env1_norm - np.mean(env1_norm)
        env2_centered = env2_norm - np.mean(env2_norm)
        norm1 = np.linalg.norm(env1_centered)
        norm2 = np.linalg.norm(env2_centered)

        if norm1 > 0 and norm2 > 0:
            envelope_similarity = np.dot(env1_centered, env2_centered) / (norm1 * norm2)
            envelope_similarity = max(0.0, envelope_similarity)
        else:
            envelope_similarity = 0.0

        # === 2. 频谱带宽相似度 ===
        # 检测频谱带宽的时域变化模式是否匹配
        bw1 = feature1.spectral_bandwidth
        bw2 = feature2.spectral_bandwidth

        bw1_centered = bw1 - np.mean(bw1)
        bw2_centered = bw2 - np.mean(bw2)
        norm1 = np.linalg.norm(bw1_centered)
        norm2 = np.linalg.norm(bw2_centered)

        if norm1 > 0 and norm2 > 0:
            bandwidth_similarity = np.dot(bw1_centered, bw2_centered) / (norm1 * norm2)
            bandwidth_similarity = max(0.0, bandwidth_similarity)
        else:
            bandwidth_similarity = 0.0

        # === 3. 频谱形状相似度 ===
        f1 = feature1.data.flatten()
        f2 = feature2.data.flatten()

        f1_centered = f1 - np.mean(f1)
        f2_centered = f2 - np.mean(f2)
        norm1 = np.linalg.norm(f1_centered)
        norm2 = np.linalg.norm(f2_centered)

        if norm1 > 0 and norm2 > 0:
            spectral_similarity = np.dot(f1_centered, f2_centered) / (norm1 * norm2)
            spectral_similarity = max(0.0, spectral_similarity)
        else:
            spectral_similarity = 0.0

        # === 4. 尖峰突出度匹配 ===
        # 如果模板有明显尖峰，测试信号也应该有
        prominence1 = feature1.peak_prominence
        prominence2 = feature2.peak_prominence

        # 计算突出度比值（期望接近1）
        if prominence1 > 1.0:  # 模板有尖峰
            prominence_ratio = min(prominence2 / prominence1, prominence1 / prominence2)
            prominence_match = prominence_ratio**0.5  # 开方使其更宽容
        else:
            prominence_match = 1.0  # 模板无明显尖峰，不惩罚

        # === 5. 综合得分 ===
        # 时域包络权重最高，因为尖峰检测最重要
        # 频谱带宽次之，用于区分全频谱和窄带信号
        # 频谱形状权重降低，避免过拟合
        envelope_weight = 0.5
        bandwidth_weight = 0.25
        spectral_weight = 0.25

        weighted_score = (
            envelope_weight * envelope_similarity
            + bandwidth_weight * bandwidth_similarity
            + spectral_weight * spectral_similarity
        )

        # 应用尖峰突出度惩罚和音量惩罚
        final_score = float(weighted_score * prominence_match * volume_penalty)

        return final_score

    def match(self, test_data: np.ndarray, start_from_sample: int = 0) -> MatchResult:
        """在测试数据中匹配模板

        Args:
            test_data: 测试音频数据
            start_from_sample: 从该采样点开始匹配（之前的数据不参与匹配）
        """
        total_duration_ms = len(test_data) / self.sample_rate * 1000

        # 计算滑动窗口参数
        window_samples = int(self.window_ms * self.sample_rate / 1000)
        step_samples = int(self.step_ms * self.sample_rate / 1000)

        # 存储所有候选
        all_candidates: List[MatchCandidate] = []

        # 从指定位置开始滑动窗口匹配
        start_sample = start_from_sample
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
    """测试数据加载和预处理

    trim_ms: 从该时间点开始匹配，之前的音频不参与匹配但会保留用于绘图
    """

    def __init__(
        self,
        sounds_dir: Path = SOUNDS_DIR,
        trim_ms: int = TEST_WAV_TRIM_MS,
        target_sample_rate: int = UNIFIED_SAMPLE_RATE,
        filter_type: str = "none",
        filter_freq: float = None,
    ):
        self.sounds_dir = sounds_dir
        self.trim_ms = trim_ms
        self.target_sample_rate = target_sample_rate
        self.unifier = AudioDataSamplesUnifier(
            target_sample_rate, filter_type=filter_type, filter_freq=filter_freq
        )

    def scan_test_files(self) -> List[Path]:
        """扫描测试WAV文件"""
        test_files = []

        if not self.sounds_dir.exists():
            logger.warn(f"测试目录不存在: {self.sounds_dir}")
            return test_files

        # 递归查找所有WAV文件
        for wav_file in self.sounds_dir.rglob("*.wav"):
            test_files.append(wav_file)

        logger.okay(f"扫描到 {len(test_files)} 个测试文件")
        return test_files

    def get_trim_samples(self) -> int:
        """获取trim对应的采样点数（在目标采样率下）"""
        return int(self.trim_ms * self.target_sample_rate / 1000)

    def load_test_file(self, file_path: Path) -> Tuple[np.ndarray, int, int]:
        """加载单个测试文件

        返回: (完整音频数据, 采样率, 开始匹配的采样点位置)
        """
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

            # 统一采样率（不截断数据）
            data = self.unifier.unify_single(sample_rate, data)

            # 计算开始匹配的采样点位置
            match_start_sample = self.get_trim_samples()
            if match_start_sample >= len(data):
                match_start_sample = 0
                logger.warn(f"trim_ms ({self.trim_ms}ms) 超过音频长度，从头开始匹配")

            return data, self.target_sample_rate, match_start_sample

        except Exception as e:
            logger.warn(f"加载测试文件失败: {file_path}, 错误: {e}")
            return np.array([]), self.target_sample_rate, 0


class MatchResultsPlotter:
    """匹配结果绘制

    绘制频谱图(spectrogram)显示从低频到高频的信号强度分布，
    并标记匹配开始位置（trim_ms）和匹配结果区域
    """

    def __init__(
        self,
        sample_rate: int = UNIFIED_SAMPLE_RATE,
        n_fft: int = 512,  # 减小n_fft以提高绘制速度
        f_max: float = None,  # None表示使用奈奎斯特频率(sample_rate/2)
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        # 如果f_max为None，使用奈奎斯特频率
        self.f_max = f_max if f_max is not None else sample_rate / 2

    def plot(
        self,
        test_data: np.ndarray,
        match_result: MatchResult,
        output_path: Path,
        title: str = "Audio Template Matching Result",
        match_start_sample: int = 0,
    ):
        """绘制匹配结果（频谱图）

        Args:
            test_data: 完整音频数据
            match_result: 匹配结果
            output_path: 输出路径
            title: 图表标题
            match_start_sample: 开始匹配的采样点位置，用于绘制分界线
        """
        import matplotlib.pyplot as plt

        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 6))

        # 计算STFT用于绘制频谱图
        hop_length = self.n_fft // 4
        frequencies, times, Zxx = signal.stft(
            test_data,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - hop_length,
        )

        # 取对数幅度谱
        magnitude = np.abs(Zxx)
        log_magnitude = 10 * np.log10(magnitude + 1e-10)

        # 限制频率范围
        freq_mask = frequencies <= self.f_max
        frequencies = frequencies[freq_mask]
        log_magnitude = log_magnitude[freq_mask, :]

        # 时间轴转换为秒
        times_s = times

        # 绘制频谱图（使用'auto'替代'gouraud'以提高速度）
        mesh = ax.pcolormesh(
            times_s,
            frequencies,
            log_magnitude,
            shading="auto",
            cmap="viridis",
        )

        # 添加颜色条
        cbar = plt.colorbar(mesh, ax=ax, label="Magnitude (dB)")

        # 绘制匹配开始位置的垂直线
        if match_start_sample > 0:
            match_start_s = match_start_sample / self.sample_rate
            ax.axvline(
                x=match_start_s,
                color="cyan",
                linewidth=2,
                linestyle="--",
                label=f"Match start ({match_start_s:.1f}s)",
            )

        # 绘制匹配区域
        y_max = frequencies[-1]

        for i, candidate in enumerate(match_result.candidates):
            # 将毫秒转换为秒
            start_s = candidate.start_ms / 1000
            end_s = candidate.end_ms / 1000
            # 绘制矩形框（半透明填充 + 细边框）
            rect = plt.Rectangle(
                (start_s, 0),
                end_s - start_s,
                y_max,
                fill=True,
                facecolor="red",
                edgecolor="darkred",
                linewidth=1,
                linestyle="-",
                alpha=0.3,
            )
            ax.add_patch(rect)

            # 添加分数文本
            text_x = (start_s + end_s) / 2
            text_y = y_max * 0.9
            ax.text(
                text_x,
                text_y,
                f"{candidate.score:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            )

        # 设置标签和标题
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right")

        # 设置X轴刻度精度为1秒
        total_duration_s = len(test_data) / self.sample_rate
        # 计算合适的刻度间隔
        tick_interval = 1  # 1秒
        x_ticks = np.arange(0, total_duration_s + tick_interval, tick_interval)
        ax.set_xticks(x_ticks)

        # 添加匹配信息
        info_text = f"Total matches: {len(match_result.candidates)}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

        # 保存图像（降低dpi以提高速度）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)


class FeatureMatchTester:
    """音频特征匹配测试"""

    def __init__(
        self,
        jsons_dir: Path = TEST_JSONS_DIR,
        plots_dir: Path = TEST_PLOTS_DIR,
        trim_ms: int = TEST_WAV_TRIM_MS,
        window_ms: float = MATCH_WINDOW_MS,
        step_ms: float = MATCH_STEP_MS,
        gate: float = MATCH_GATE,
        min_offset_ms: float = CANDIDATE_MIN_OFFSET_MS,
        filter_type: str = "none",
        filter_freq: float = None,
    ):
        self.jsons_dir = jsons_dir
        self.plots_dir = plots_dir
        self.trim_ms = trim_ms
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.gate = gate
        self.min_offset_ms = min_offset_ms
        self.filter_type = filter_type
        self.filter_freq = filter_freq

        # 初始化组件
        self.template_loader = TemplateLoader(TEMPLATES_DIR, TEMPLATE_REGEX)
        self.unifier = AudioDataSamplesUnifier(
            filter_type=filter_type, filter_freq=filter_freq
        )
        self.feature_extractor = FeatureExtractor()
        self.aligner = TemplateAligner()
        self.fuser = TemplateFuser(self.feature_extractor)
        self.test_loader = TestDataSamplesLoader(
            SOUNDS_DIR, trim_ms, filter_type=filter_type, filter_freq=filter_freq
        )
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

        # 找到中心融合区域（保证至少600ms）
        region_start, region_end = self.aligner.find_center_region(
            unified_templates, offsets
        )

        # 提取对齐后的区域（带覆盖率权重）
        extracted_templates, weights = self.aligner.extract_aligned_region(
            unified_templates, offsets, region_start, region_end
        )

        # 输出每个模板的覆盖率
        avg_coverage = sum(weights) / len(weights) if weights else 0
        logger.mesg(f"模板平均覆盖率: {avg_coverage:.1%}")

        # 融合模板（使用覆盖率作为权重）
        logger.hint("融合模板特征...")
        self.fused_template_feature = self.fuser.fuse_templates(
            extracted_templates, weights
        )

        # 根据模板时长更新匹配窗口大小
        self.window_ms = self.fused_template_feature.duration_ms
        logger.okay(
            f"模板特征融合完成, 特征维度: {self.fused_template_feature.data.shape}, "
            f"模板时长: {self.window_ms:.1f}ms"
        )

        return self.fused_template_feature

        return self.fused_template_feature

    def get_output_filename(self, test_file: Path) -> str:
        """生成输出文件名"""
        parent_name = test_file.parent.name
        file_stem = test_file.stem
        return f"{parent_name}_{file_stem}"

    def test_single_file(
        self, test_file: Path, file_index: int = 0, total_files: int = 1
    ) -> MatchResult:
        """测试单个文件"""
        # 加载测试数据（返回完整数据和开始匹配的位置）
        test_data, sample_rate, match_start_sample = self.test_loader.load_test_file(
            test_file
        )

        if len(test_data) == 0:
            logger.warn(f"[{file_index}/{total_files}] {test_file.name} - 数据为空")
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

        # 执行匹配（从指定位置开始），记录匹配时长
        match_start_time = time.time()
        result = matcher.match(test_data, start_from_sample=match_start_sample)
        match_duration = time.time() - match_start_time
        result.test_file = str(test_file)

        # 根据匹配结果输出日志
        num_candidates = len(result.candidates)
        if num_candidates > 0:
            max_score = max(c.score for c in result.candidates)
            logger.okay(
                f"[{file_index}/{total_files}] {test_file.name} - "
                f"候选数：{num_candidates}，最高分数：{max_score:.4f}，"
                f"匹配耗时：{match_duration:.2f}秒"
            )
        else:
            logger.warn(
                f"[{file_index}/{total_files}] {test_file.name} - "
                f"候选数：0，匹配耗时：{match_duration:.2f}秒"
            )

        # 生成输出文件名
        output_name = self.get_output_filename(test_file)

        # 保存JSON结果
        json_path = self.jsons_dir / f"{output_name}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # 绘制并保存图像（传入match_start_sample用于绘制分界线）
        plot_path = self.plots_dir / f"{output_name}.jpg"
        self.plotter.plot(
            test_data,
            result,
            plot_path,
            title=f"Match Result: {test_file.name}",
            match_start_sample=match_start_sample,
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
        total_files = len(test_files)
        for i, test_file in enumerate(test_files, 1):
            result = self.test_single_file(
                test_file, file_index=i, total_files=total_files
            )
            results.append(result)

        logger.hint("=" * 50)
        logger.okay(f"测试完成, 共处理 {len(results)} 个文件")

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
        self.parser.add_argument(
            "--filter-type",
            type=str,
            choices=["none", "lowpass", "highpass"],
            default="none",
            help="滤波器类型: none(不滤波), lowpass(低通), highpass(高通)",
        )
        self.parser.add_argument(
            "--filter-freq",
            type=float,
            default=None,
            help="滤波截止频率(Hz)，默认低通6000Hz/高通10000Hz",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数"""
        return self.parser.parse_args()


def main():
    """主函数"""
    arg_parser = TemplateMatcherArgParser()
    args = arg_parser.parse()

    if args.test:
        logger.okay("启动音频模板匹配测试...")
        if args.filter_type != "none":
            if args.filter_type == "lowpass":
                default_hz = LOW_PASS_CUTOFF_HZ
            elif args.filter_type == "highpass":
                default_hz = HIGH_PASS_CUTOFF_HZ
            else:
                default_hz = "未知"
            logger.mesg(
                f"滤波器: {args.filter_type}, "
                f"截止频率: {args.filter_freq or default_hz}Hz"
            )

        tester = FeatureMatchTester(
            trim_ms=args.trim_ms,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            gate=args.gate,
            min_offset_ms=args.min_offset_ms,
            filter_type=args.filter_type,
            filter_freq=args.filter_freq,
        )

        results = tester.test_all_files()

        # 输出摘要
        total_candidates = sum(len(r.candidates) for r in results)
        logger.hint("=" * 50)
        logger.okay(f"测试摘要:")
        logger.mesg(f"  - 测试文件数: {len(results)}")
        logger.mesg(f"  - 总匹配数: {total_candidates}")
    else:
        arg_parser.parser.print_help()


if __name__ == "__main__":
    main()

    # python -m gtaz.audios.signals_v3 --test
    # python -m gtaz.audios.signals_v3 --test --filter-type lowpass
    # python -m gtaz.audios.signals_v3 --test --filter-type highpass
