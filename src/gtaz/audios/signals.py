"""音频模板匹配模块

安装依赖：

```sh
pip install scipy
```
"""

import argparse
import json
import numpy as np
import soundfile as sf
import threading
import time

from dataclasses import dataclass, field
from numpy.fft import rfft, irfft
from pathlib import Path
from scipy import signal as scipy_signal
from typing import Optional, Callable, Any
from tclogger import TCLogger, logstr, dict_to_lines

from .sounds import (
    SoundRecorder,
    AudioBuffer,
    AUDIO_DEVICE_NAME,
    SAMPLE_RATE,
    CHANNELS,
    WINDOW_MS,
)


logger = TCLogger(
    name="TemplateMatcher",
    use_prefix=False,
    use_file=True,
    file_path=Path(__file__).parent / "output.log",
    file_mode="w",
)


# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR.parent / "cache"
# 模板目录（默认在 wavs 子目录中）
TEMPLATES_DIR = MODULE_DIR / "wavs"
# 模板文件前缀
TEMPLATE_PREFIX = "template_"
# 测试音频目录
TEST_SOUNDS_DIR = CACHE_DIR / "sounds"

# 默认匹配阈值（0-1，越高表示匹配度越高）
MATCH_THRESHOLD = 0.82
# 默认检测冷却时间（毫秒）
COOLDOWN_MS = 2000
# 实时匹配的目标采样率（降采样以提高速度）
REALTIME_SAMPLE_RATE = 16000


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
class AudioTemplate:
    """
    音频模板数据类。

    存储音频模板的元信息和波形数据。
    """

    name: str
    """模板名称"""
    data: np.ndarray
    """音频数据 (samples,) 或 (samples, channels)"""
    sample_rate: int
    """采样率"""
    file_path: Optional[Path] = None
    """源文件路径"""

    @property
    def duration_ms(self) -> float:
        """获取模板时长（毫秒）。"""
        return len(self.data) / self.sample_rate * 1000

    @property
    def samples(self) -> int:
        """获取采样点数。"""
        return len(self.data)

    @property
    def channels(self) -> int:
        """获取通道数。"""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[1]

    def to_mono(self) -> np.ndarray:
        """
        转换为单声道。

        :return: 单声道音频数据 (samples,)
        """
        if self.data.ndim == 1:
            return self.data
        return np.mean(self.data, axis=1)

    def __repr__(self) -> str:
        return (
            f"AudioTemplate("
            f"name={self.name}, "
            f"samples={self.samples}, "
            f"duration={self.duration_ms:.1f}ms, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels})"
        )


class TemplateLoader:
    """
    模板加载器。

    从文件或目录加载音频模板。
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        """
        初始化模板加载器。

        :param templates_dir: 模板目录
        """
        self.templates_dir = Path(templates_dir)
        self._templates: dict[str, AudioTemplate] = {}

    def load_file(self, file_path: Path, name: str = None) -> Optional[AudioTemplate]:
        """
        从文件加载单个模板。

        :param file_path: 音频文件路径
        :param name: 模板名称，默认使用文件名（不含扩展名）

        :return: 音频模板对象，失败则返回 None
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warn(f"模板文件不存在: {file_path}")
            return None

        try:
            data, sample_rate = sf.read(file_path)
            if name is None:
                name = file_path.stem

            template = AudioTemplate(
                name=name,
                data=data,
                sample_rate=sample_rate,
                file_path=file_path,
            )
            self._templates[name] = template
            # logger.okay(f"已加载模板: {template}")
            return template
        except Exception as e:
            logger.warn(f"加载模板文件失败 {file_path}: {e}")
            return None

    def load_directory(
        self,
        dir_path: Path = None,
        extensions: list[str] = None,
        prefix: str = TEMPLATE_PREFIX,
    ) -> int:
        """
        从目录加载所有模板。

        :param dir_path: 目录路径，默认使用 templates_dir
        :param extensions: 支持的文件扩展名，默认 [".wav", ".flac", ".ogg", ".mp3"]
        :param prefix: 文件名前缀过滤，默认 "template_"，设为 None 或空字符串则不过滤

        :return: 成功加载的模板数量
        """
        if dir_path is None:
            dir_path = self.templates_dir
        dir_path = Path(dir_path)

        if extensions is None:
            extensions = [".wav", ".flac", ".ogg", ".mp3"]

        if not dir_path.exists():
            logger.warn(f"模板目录不存在: {dir_path}")
            return 0

        count = 0
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # 检查文件名前缀
                if prefix and not file_path.name.startswith(prefix):
                    continue
                if self.load_file(file_path):
                    count += 1

        logger.note(f"从目录 {dir_path} 加载了 {count} 个模板")
        return count

    def get_template(self, name: str) -> Optional[AudioTemplate]:
        """
        获取指定名称的模板。

        :param name: 模板名称

        :return: 音频模板对象，不存在则返回 None
        """
        return self._templates.get(name)

    def get_all_templates(self) -> list[AudioTemplate]:
        """
        获取所有已加载的模板。

        :return: 模板列表
        """
        return list(self._templates.values())

    def clear(self):
        """清空所有模板。"""
        self._templates.clear()

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"TemplateLoader(templates_dir={self.templates_dir}, count={len(self)})"


@dataclass
class AudioFeatures:
    """
    音频特征数据类。

    存储音频特征提取的结果，包括处理后的数据、能量等信息。
    """

    data: np.ndarray
    """处理后的音频数据（单声道，float32）"""
    sample_rate: int
    """采样率"""
    energy: float = 0.0
    """音频能量（零均值后的平方和）"""
    mean: float = 0.0
    """音频均值"""
    centered_data: Optional[np.ndarray] = None
    """零均值化后的数据（可选，用于优化）"""

    # 新增特征
    spectrogram: Optional[np.ndarray] = None
    """频谱图 (频率bins, 时间帧)"""
    mel_spectrogram: Optional[np.ndarray] = None
    """梅尔频谱图"""
    energy_envelope: Optional[np.ndarray] = None
    """能量包络"""
    zero_crossing_rate: float = 0.0
    """零交叉率"""
    spectral_centroid: Optional[np.ndarray] = None
    """频谱质心"""

    @property
    def samples(self) -> int:
        """获取采样点数。"""
        return len(self.data)

    @property
    def duration_ms(self) -> float:
        """获取音频时长（毫秒）。"""
        return self.samples / self.sample_rate * 1000

    def __repr__(self) -> str:
        return (
            f"AudioFeatures("
            f"samples={self.samples}, "
            f"sample_rate={self.sample_rate}, "
            f"energy={self.energy:.2f}, "
            f"mean={self.mean:.4f})"
        )


class FeaturesExtractor:
    """
    音频特征提取器。

    负责音频预处理和特征提取，包括：
    - 单声道转换
    - 重采样
    - 数据类型转换
    - 能量计算
    - 零均值化
    - 频谱特征（STFT、梅尔频谱）
    - 能量包络
    - 零交叉率
    - 频谱质心
    """

    def __init__(self, target_sample_rate: int = REALTIME_SAMPLE_RATE):
        """
        初始化特征提取器。

        :param target_sample_rate: 目标采样率
        """
        self.target_sample_rate = target_sample_rate

        # STFT 参数（优化：减小窗口和增加hop以提升速度）
        self.n_fft = 256  # FFT 窗口大小（从512降至256）
        self.hop_length = 256  # 帧移（从128增至256，减少帧数）
        self.n_mels = 20  # 梅尔滤波器数量（从40降至20）

    @staticmethod
    def to_mono(data: np.ndarray) -> np.ndarray:
        """
        转换为单声道。

        :param data: 音频数据 (samples,) 或 (samples, channels)

        :return: 单声道数据 (samples,)
        """
        if data.ndim == 1:
            return data
        return np.mean(data, axis=1)

    @staticmethod
    def resample(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """
        重采样音频数据。

        :param data: 音频数据
        :param from_rate: 原始采样率
        :param to_rate: 目标采样率

        :return: 重采样后的数据
        """
        if from_rate == to_rate:
            return data

        num_samples = int(len(data) * to_rate / from_rate)
        return scipy_signal.resample(data, num_samples)

    def preprocess(
        self,
        data: np.ndarray,
        sample_rate: int,
        to_mono: bool = True,
        resample: bool = True,
    ) -> np.ndarray:
        """
        预处理音频数据。

        :param data: 原始音频数据
        :param sample_rate: 原始采样率
        :param to_mono: 是否转换为单声道
        :param resample: 是否重采样

        :return: 预处理后的数据（float32）
        """
        # 转单声道
        if to_mono:
            if data.ndim > 1:
                data = self.to_mono(data)
            else:
                data = data.copy()
        else:
            data = data.copy()

        # 重采样
        if resample and sample_rate != self.target_sample_rate:
            data = self.resample(data, sample_rate, self.target_sample_rate)

        # 转换为 float32
        return data.astype(np.float32)

    @staticmethod
    def compute_energy(data: np.ndarray, zero_mean: bool = True) -> float:
        """
        计算音频能量。

        :param data: 音频数据
        :param zero_mean: 是否先减去均值

        :return: 能量值（平方和）
        """
        if zero_mean:
            data_mean = np.mean(data)
            data_centered = data - data_mean
            return float(np.sum(data_centered**2))
        return float(np.sum(data**2))

    @staticmethod
    def compute_zero_crossing_rate(data: np.ndarray) -> float:
        """
        计算零交叉率。

        :param data: 音频数据

        :return: 零交叉率（0-1）
        """
        if len(data) < 2:
            return 0.0
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
        return float(zero_crossings / (len(data) - 1))

    def compute_energy_envelope(
        self, data: np.ndarray, frame_length: int = None
    ) -> np.ndarray:
        """
        计算能量包络。

        :param data: 音频数据
        :param frame_length: 帧长度，默认使用 hop_length

        :return: 能量包络数组
        """
        if frame_length is None:
            frame_length = self.hop_length

        # 计算每帧的 RMS 能量
        num_frames = (len(data) - frame_length) // frame_length + 1
        envelope = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * frame_length
            end = start + frame_length
            frame = data[start:end]
            envelope[i] = np.sqrt(np.mean(frame**2))

        return envelope

    def compute_spectrogram(self, data: np.ndarray) -> np.ndarray:
        """
        计算频谱图（STFT）- 向量化优化版本。

        :param data: 音频数据

        :return: 频谱图幅度 (频率bins, 时间帧)
        """
        # 使用汉宁窗
        window = np.hanning(self.n_fft)

        # 计算帧数
        num_frames = (len(data) - self.n_fft) // self.hop_length + 1
        if num_frames <= 0:
            return np.array([[]])

        # 向量化构建所有帧（避免循环）
        # 创建索引矩阵
        indices = (
            np.arange(self.n_fft)[:, np.newaxis]
            + np.arange(num_frames) * self.hop_length
        )

        # 一次性提取所有帧
        frames = data[indices] * window[:, np.newaxis]

        # 对所有帧进行FFT（向量化操作）
        fft_result = rfft(frames, axis=0)
        spectrogram = np.abs(fft_result)

        # 转换为对数尺度（避免 log(0)）
        spectrogram = np.log(spectrogram + 1e-10)

        return spectrogram

    def compute_mel_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        计算梅尔频谱图 - 优化版本。

        :param spectrogram: 线性频谱图

        :return: 梅尔频谱图
        """
        n_freqs = spectrogram.shape[0]

        # 简化版梅尔滤波器（线性插值近似）
        # 创建mel频段的边界索引
        mel_points = np.linspace(0, n_freqs - 1, self.n_mels + 2, dtype=int)

        # 使用简单的分段求和替代三角滤波器
        mel_spec = np.zeros((self.n_mels, spectrogram.shape[1]))

        for i in range(self.n_mels):
            start = mel_points[i]
            end = mel_points[i + 2]
            # 直接对频段求平均（比三角滤波快很多）
            mel_spec[i] = np.mean(spectrogram[start:end], axis=0)

        return mel_spec

    def compute_spectral_centroid(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        计算频谱质心。

        :param spectrogram: 频谱图

        :return: 频谱质心数组（每帧一个值）
        """
        # 计算频率bins
        freqs = np.linspace(0, self.target_sample_rate / 2, spectrogram.shape[0])

        # 计算每帧的频谱质心
        magnitude = np.exp(spectrogram)  # 转回线性尺度
        centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / (
            np.sum(magnitude, axis=0) + 1e-10
        )

        return centroid

    def extract(
        self,
        data: np.ndarray,
        sample_rate: int,
        compute_centered: bool = False,
        compute_spectral: bool = True,
    ) -> AudioFeatures:
        """
        提取音频特征。

        :param data: 原始音频数据
        :param sample_rate: 原始采样率
        :param compute_centered: 是否计算零均值化数据
        :param compute_spectral: 是否计算频谱特征

        :return: 音频特征对象
        """
        # 预处理
        processed_data = self.preprocess(data, sample_rate)

        # 计算均值
        data_mean = float(np.mean(processed_data))

        # 计算能量
        energy = self.compute_energy(processed_data, zero_mean=True)

        # 零均值化（可选）
        centered_data = None
        if compute_centered:
            centered_data = processed_data - data_mean

        # 计算频谱特征
        spectrogram = None
        mel_spec = None
        spectral_centroid = None
        if compute_spectral and len(processed_data) >= self.n_fft:
            spectrogram = self.compute_spectrogram(processed_data)
            if spectrogram.size > 0:
                mel_spec = self.compute_mel_spectrogram(spectrogram)
                spectral_centroid = self.compute_spectral_centroid(spectrogram)

        # 计算能量包络
        energy_envelope = None
        if len(processed_data) >= self.hop_length:
            energy_envelope = self.compute_energy_envelope(processed_data)

        # 计算零交叉率
        zcr = self.compute_zero_crossing_rate(processed_data)

        return AudioFeatures(
            data=processed_data,
            sample_rate=self.target_sample_rate,
            energy=energy,
            mean=data_mean,
            centered_data=centered_data,
            spectrogram=spectrogram,
            mel_spectrogram=mel_spec,
            energy_envelope=energy_envelope,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
        )

    def extract_from_template(self, template: AudioTemplate) -> AudioFeatures:
        """
        从音频模板提取特征。

        :param template: 音频模板

        :return: 音频特征对象
        """
        mono_data = template.to_mono()
        return self.extract(
            mono_data,
            template.sample_rate,
            compute_centered=True,  # 模板需要零均值化数据用于匹配
        )

    def __repr__(self) -> str:
        return f"FeaturesExtractor(target_sample_rate={self.target_sample_rate})"


@dataclass
class MatchResult:
    """
    匹配结果数据类。

    存储单次匹配的结果信息。
    """

    template_name: str
    """模板名称"""
    score: float
    """匹配分数（0-1）"""
    matched: bool
    """是否匹配成功"""
    timestamp: float
    """匹配时间戳"""
    position: int = 0
    """匹配位置（采样点索引，基于 target_sample_rate）"""
    confidence: float = 0.0
    """置信度（基于分数计算）"""
    sample_rate: int = 0
    """匹配时使用的采样率（target_sample_rate，如 16000Hz）"""
    original_sample_rate: int = 0
    """原始音频的采样率（如 44100Hz）"""
    template_duration_ms: float = 0.0
    """模板时长（毫秒）"""

    def to_dict(self) -> dict:
        """转换为字典，用于 JSON 序列化。"""
        # 计算匹配位置的时间（基于 position 所在的采样率）
        # 注意：多片段检测可能已将音频重采样到 target_sample_rate，此时 position
        # 已经是 target_sample_rate 域的绝对采样点索引；不应再转换回 original_sample_rate。
        if self.sample_rate > 0:
            start_seconds = self.position / self.sample_rate
            duration_seconds = self.template_duration_ms / 1000
            end_seconds = start_seconds + duration_seconds

            # 格式化为 SS.sss
            start_time = f"{start_seconds:.3f}"
            end_time = f"{end_seconds:.3f}"
            duration = f"{duration_seconds:.3f}"
        else:
            start_time = "N/A"
            end_time = "N/A"
            duration = "N/A"

        return {
            "template_name": self.template_name,
            "score": round(self.score, 4),
            "matched": self.matched,
            "timestamp": self.timestamp,
            "position": self.position,
            "confidence": round(self.confidence, 4),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
        }

    def __repr__(self) -> str:
        return (
            f"MatchResult("
            f"template={self.template_name}, "
            f"score={self.score:.3f}, "
            f"matched={self.matched}, "
            f"confidence={self.confidence:.3f})"
        )


class TemplateMatcher:
    """
    音频模板匹配器（多特征融合版）。

    使用多种音频特征进行模板匹配，提高泛化能力：
    - 时域相关性（NCC）
    - 频谱相似度（谱图、梅尔谱图）
    - 能量包络匹配
    - 零交叉率相似度
    - 频谱质心相似度
    通过加权融合多个特征的匹配分数，实现更鲁棒的音频匹配。
    """

    def __init__(
        self,
        templates: list[AudioTemplate] = None,
        threshold: float = MATCH_THRESHOLD,
        target_sample_rate: int = REALTIME_SAMPLE_RATE,
        cooldown_ms: float = COOLDOWN_MS,
        features_extractor: FeaturesExtractor = None,
        # 特征权重（提高频谱权重，降低时域权重以减少假阳性）
        weight_time_corr: float = 0.15,
        weight_mel_spec: float = 0.45,
        weight_energy_env: float = 0.30,
        weight_zcr: float = 0.0,
        weight_spectral_centroid: float = 0.10,
    ):
        """
        初始化模板匹配器。

        :param templates: 音频模板列表
        :param threshold: 匹配阈值（0-1）
        :param target_sample_rate: 目标采样率（用于重采样，较低采样率可提高速度）
        :param cooldown_ms: 检测冷却时间（毫秒）
        :param features_extractor: 特征提取器，默认自动创建
        :param weight_time_corr: 时域相关性权重
        :param weight_mel_spec: 梅尔频谱权重
        :param weight_energy_env: 能量包络权重
        :param weight_zcr: 零交叉率权重
        :param weight_spectral_centroid: 频谱质心权重
        """
        self.threshold = threshold
        self.target_sample_rate = target_sample_rate
        self.cooldown_ms = cooldown_ms

        # 特征权重（归一化）
        total_weight = (
            weight_time_corr
            + weight_mel_spec
            + weight_energy_env
            + weight_zcr
            + weight_spectral_centroid
        )
        self.weight_time_corr = weight_time_corr / total_weight
        self.weight_mel_spec = weight_mel_spec / total_weight
        self.weight_energy_env = weight_energy_env / total_weight
        self.weight_zcr = weight_zcr / total_weight
        self.weight_spectral_centroid = weight_spectral_centroid / total_weight

        # 特征提取器
        self.features_extractor = features_extractor or FeaturesExtractor(
            target_sample_rate=target_sample_rate
        )

        # 原始模板引用
        self._templates: dict[str, AudioTemplate] = {}
        # 预处理后的模板特征
        self._template_features: dict[str, AudioFeatures] = {}
        # 上次匹配时间（用于冷却）
        self._last_match_times: dict[str, float] = {}

        if templates:
            for template in templates:
                self.add_template(template)

    def add_template(self, template: AudioTemplate):
        """
        添加模板。

        :param template: 音频模板
        """
        # 使用特征提取器提取模板特征
        features = self.features_extractor.extract_from_template(template)

        # 保存原始模板和处理后的特征
        self._templates[template.name] = template
        self._template_features[template.name] = features
        self._last_match_times[template.name] = 0

        logger.note(
            f"- 已添加模板: {template.name} "
            f"({len(features.data)} samples @ {features.sample_rate}Hz)"
        )

    def remove_template(self, name: str):
        """
        移除模板。

        :param name: 模板名称
        """
        if name in self._templates:
            del self._templates[name]
            del self._template_features[name]
            del self._last_match_times[name]

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化音频数据。

        :param data: 音频数据

        :return: 归一化后的数据
        """
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    def _is_in_cooldown(self, template_name: str, current_time: float) -> bool:
        """
        判断是否处于冷却期。

        :param template_name: 模板名称
        :param current_time: 当前时间戳

        :return: 是否处于冷却期
        """
        last_time = self._last_match_times.get(template_name, 0)
        elapsed_ms = (current_time - last_time) * 1000
        return elapsed_ms < self.cooldown_ms

    def _compute_correlation_fft(
        self, window_features: AudioFeatures, template_features: AudioFeatures
    ) -> tuple[float, int]:
        """
        使用 FFT 计算归一化互相关（快速版本）。

        :param window_features: 窗口音频特征
        :param template_features: 模板音频特征

        :return: (最大相关系数, 最佳匹配位置)
        """
        window = window_features.centered_data
        template_zm = template_features.centered_data
        template_energy = template_features.energy

        window_len = len(window)
        template_len = len(template_zm)

        # 确保窗口长度大于模板
        if window_len < template_len:
            return 0.0, 0

        if template_energy <= 0:
            return 0.0, 0

        # 计算 FFT 长度（选择 2 的幂次以加速）
        fft_len = 1
        while fft_len < window_len + template_len - 1:
            fft_len *= 2

        # 预计算模板的 FFT（已零均值化）
        template_zm_fft = rfft(template_zm[::-1], fft_len)

        # 计算窗口的 FFT
        window_fft = rfft(window, fft_len)

        # 计算窗口与零均值模板的互相关
        correlation = irfft(window_fft * template_zm_fft, fft_len)

        # 只取有效部分 (valid mode)
        valid_start = template_len - 1
        valid_len = window_len - template_len + 1
        correlation = correlation[valid_start : valid_start + valid_len]

        # 计算窗口局部均值并修正相关值
        # 相关值需要减去 window_mean * sum(template_zm)，但 sum(template_zm) = 0
        # 所以不需要修正！这是使用零均值模板的好处

        # 计算窗口局部能量（使用累积和优化）
        window_sq = window**2
        cumsum_sq = np.concatenate(([0], np.cumsum(window_sq)))
        window_sq_sum = cumsum_sq[template_len:] - cumsum_sq[:-template_len]

        # 计算窗口局部均值
        cumsum_val = np.concatenate(([0], np.cumsum(window)))
        window_sum = cumsum_val[template_len:] - cumsum_val[:-template_len]
        window_mean = window_sum / template_len

        # 计算窗口局部方差能量: sum((x - mean)^2) = sum(x^2) - n * mean^2
        window_std_energy = window_sq_sum - template_len * window_mean**2

        # 确保方差能量非负
        window_std_energy = np.maximum(window_std_energy, 0.0)

        # 修正相关值：减去窗口均值的影响
        # corrected_correlation = sum(window * template_zm) - window_mean * sum(template_zm)
        # 由于 sum(template_zm) = 0，所以 corrected_correlation = correlation
        # 但实际上我们计算的是 sum(window * template_zm)，而不是 sum((window - window_mean) * template_zm)
        # 需要修正：sum((window - window_mean) * template_zm) = sum(window * template_zm) - window_mean * sum(template_zm)
        #                                                     = correlation - window_mean * 0 = correlation
        # 所以这里不需要修正！

        # 计算归一化系数
        denominator = np.sqrt(window_std_energy * template_energy)

        # 设置最小能量阈值：窗口能量至少是模板能量的一定比例
        # 这可以过滤掉低能量（静音）区域的假匹配
        min_energy_ratio = 0.15  # 15% 平衡过滤噪声和保留真实匹配
        min_window_energy = min_energy_ratio * template_energy

        # 创建有效性掩码
        valid_mask = window_std_energy >= min_window_energy

        # 安全计算 NCC
        safe_denominator = np.where(denominator > 0, denominator, 1.0)
        normalized = np.where(valid_mask, correlation / safe_denominator, -np.inf)

        # 找到最大值及其位置
        max_idx = np.argmax(normalized)
        max_score = float(normalized[max_idx])

        if not np.isfinite(max_score):
            return 0.0, 0

        # 限制分数范围并过滤负相关
        max_score = min(max(max_score, 0.0), 1.0)

        return max_score, int(max_idx)

    def _compute_spectral_similarity(
        self,
        window_spec: np.ndarray,
        template_spec: np.ndarray,
    ) -> tuple[float, int]:
        """
        计算频谱相似度（使用向量化优化的滑动窗口）。

        :param window_spec: 窗口频谱图 (频率bins, 时间帧)
        :param template_spec: 模板频谱图 (频率bins, 时间帧)

        :return: (最大相似度, 最佳匹配位置的帧索引)
        """
        if window_spec is None or template_spec is None:
            return 0.0, 0

        if window_spec.size == 0 or template_spec.size == 0:
            return 0.0, 0

        n_freqs_w, n_frames_w = window_spec.shape
        n_freqs_t, n_frames_t = template_spec.shape

        if n_freqs_w != n_freqs_t or n_frames_w < n_frames_t:
            return 0.0, 0

        # 展平为向量以加速计算
        template_vec = template_spec.flatten()
        template_norm = np.linalg.norm(template_vec)

        if template_norm == 0:
            return 0.0, 0

        # 向量化计算所有位置的余弦相似度
        num_positions = n_frames_w - n_frames_t + 1
        similarities = np.zeros(num_positions)

        # 预计算窗口切片的归一化
        for i in range(num_positions):
            window_slice = window_spec[:, i : i + n_frames_t].flatten()
            window_norm = np.linalg.norm(window_slice)

            if window_norm > 0:
                similarities[i] = np.dot(window_slice, template_vec) / (
                    window_norm * template_norm
                )

        # 找到最大值
        best_idx = np.argmax(similarities)
        max_similarity = similarities[best_idx]

        # 将相似度限制在 [0, 1] 范围
        max_similarity = min(max(max_similarity, 0.0), 1.0)

        return float(max_similarity), int(best_idx)

    def _compute_envelope_similarity(
        self,
        window_env: np.ndarray,
        template_env: np.ndarray,
    ) -> tuple[float, int]:
        """
        计算能量包络相似度（向量化优化版本）。

        :param window_env: 窗口能量包络
        :param template_env: 模板能量包络

        :return: (最大相似度, 最佳匹配位置)
        """
        if window_env is None or template_env is None:
            return 0.0, 0

        if len(window_env) == 0 or len(template_env) == 0:
            return 0.0, 0

        window_len = len(window_env)
        template_len = len(template_env)

        if window_len < template_len:
            return 0.0, 0

        # 归一化包络（使用 L2 归一化）
        template_norm_val = np.linalg.norm(template_env)
        if template_norm_val == 0:
            return 0.0, 0

        template_norm = template_env / template_norm_val

        # 使用卷积快速计算所有位置的点积
        # 注意：需要对窗口的每个切片单独归一化，所以还是要循环
        # 但可以向量化部分计算
        num_positions = window_len - template_len + 1
        similarities = np.zeros(num_positions)

        for i in range(num_positions):
            window_slice = window_env[i : i + template_len]
            window_norm_val = np.linalg.norm(window_slice)

            if window_norm_val > 0:
                similarities[i] = np.dot(window_slice, template_norm) / window_norm_val

        # 找到最大值
        best_idx = np.argmax(similarities)
        max_similarity = similarities[best_idx]

        # 将相似度限制在 [0, 1] 范围
        max_similarity = min(max(max_similarity, 0.0), 1.0)

        return float(max_similarity), int(best_idx)

    def _compute_zcr_similarity(
        self,
        window_zcr: float,
        template_zcr: float,
    ) -> float:
        """
        计算零交叉率相似度。

        :param window_zcr: 窗口零交叉率
        :param template_zcr: 模板零交叉率

        :return: 相似度 (0-1)
        """
        # 使用高斯核计算相似度
        # ZCR 差异越小，相似度越高
        diff = abs(window_zcr - template_zcr)
        # 设置一个合理的标准差（根据经验调整）
        sigma = 0.1
        similarity = np.exp(-0.5 * (diff / sigma) ** 2)
        return float(similarity)

    def _compute_centroid_similarity(
        self,
        window_centroid: np.ndarray,
        template_centroid: np.ndarray,
    ) -> tuple[float, int]:
        """
        计算频谱质心相似度（向量化优化版本）。

        :param window_centroid: 窗口频谱质心数组
        :param template_centroid: 模板频谱质心数组

        :return: (最大相似度, 最佳匹配位置)
        """
        if window_centroid is None or template_centroid is None:
            return 0.0, 0

        if len(window_centroid) == 0 or len(template_centroid) == 0:
            return 0.0, 0

        window_len = len(window_centroid)
        template_len = len(template_centroid)

        if window_len < template_len:
            return 0.0, 0

        # 归一化质心（按照最大频率）
        max_freq = self.target_sample_rate / 2
        window_norm = window_centroid / max_freq
        template_norm = template_centroid / max_freq

        # 向量化计算所有位置的相似度
        num_positions = window_len - template_len + 1

        # 使用滑动窗口视图（避免复制数据）
        # 计算所有位置的MSE
        similarities = np.zeros(num_positions)

        for i in range(num_positions):
            window_slice = window_norm[i : i + template_len]
            mse = np.mean((window_slice - template_norm) ** 2)
            similarities[i] = np.exp(-mse * 10)  # 转换为相似度

        # 找到最大值
        best_idx = np.argmax(similarities)
        max_similarity = similarities[best_idx]

        # 将相似度限制在 [0, 1] 范围
        max_similarity = min(max(max_similarity, 0.0), 1.0)

        return float(max_similarity), int(best_idx)

    def match_single(
        self,
        window_data: np.ndarray,
        template_name: str,
        sample_rate: int = None,
        check_cooldown: bool = True,
        preprocessed_window: np.ndarray = None,
    ) -> MatchResult:
        """
        对单个模板进行多特征匹配。

        :param window_data: 窗口音频数据
        :param template_name: 模板名称
        :param sample_rate: 窗口采样率，默认使用 target_sample_rate
        :param check_cooldown: 是否检查冷却时间
        :param preprocessed_window: 已预处理的窗口特征（可选，用于批量匹配时避免重复预处理）

        :return: 匹配结果
        """
        current_time = time.time()

        # 检查模板是否存在
        if template_name not in self._template_features:
            return MatchResult(
                template_name=template_name,
                score=0.0,
                matched=False,
                timestamp=current_time,
            )

        # 检查冷却时间
        if check_cooldown and self._is_in_cooldown(template_name, current_time):
            return MatchResult(
                template_name=template_name,
                score=0.0,
                matched=False,
                timestamp=current_time,
            )

        # 获取模板特征
        template_features = self._template_features[template_name]

        # 预处理窗口数据
        if preprocessed_window is not None:
            window_features = preprocessed_window
        else:
            if sample_rate is None:
                sample_rate = self.target_sample_rate
            window_features = self.features_extractor.extract(
                window_data, sample_rate, compute_centered=True, compute_spectral=True
            )

        # 1. 计算时域相关性（使用 FFT 加速）
        time_score, time_position = self._compute_correlation_fft(
            window_features, template_features
        )

        # 2. 计算梅尔频谱相似度
        mel_score, mel_position = self._compute_spectral_similarity(
            window_features.mel_spectrogram,
            template_features.mel_spectrogram,
        )

        # 3. 计算能量包络相似度
        env_score, env_position = self._compute_envelope_similarity(
            window_features.energy_envelope,
            template_features.energy_envelope,
        )

        # 4. 计算零交叉率相似度
        zcr_score = self._compute_zcr_similarity(
            window_features.zero_crossing_rate,
            template_features.zero_crossing_rate,
        )

        # 5. 计算频谱质心相似度
        centroid_score, centroid_position = self._compute_centroid_similarity(
            window_features.spectral_centroid,
            template_features.spectral_centroid,
        )

        # 如果关键特征过低，直接判定为不匹配以削减假阳性
        mel_gate = 0.80
        env_gate = 0.78
        if mel_score < mel_gate or env_score < env_gate:
            matched = False
            final_score = 0.0
            position = 0
        else:
            # 加权融合所有特征分数
            final_score = (
                self.weight_time_corr * time_score
                + self.weight_mel_spec * mel_score
                + self.weight_energy_env * env_score
                + self.weight_zcr * zcr_score
                + self.weight_spectral_centroid * centroid_score
            )

            # 使用时域位置作为主要位置（因为它最准确）
            # 但如果时域匹配很差，可以考虑其他特征的位置
            if time_score < 0.3 and mel_score > time_score:
                # 将梅尔频谱的帧位置转换为采样点位置
                position = mel_position * self.features_extractor.hop_length
            else:
                position = time_position

            # 判断是否匹配
            matched = final_score >= self.threshold

        # 更新冷却时间
        if matched:
            self._last_match_times[template_name] = current_time

        # 计算置信度（将分数映射到 0-1 范围，以阈值为中点）
        if final_score >= self.threshold:
            confidence = 0.5 + 0.5 * (final_score - self.threshold) / (
                1 - self.threshold + 1e-10
            )
        else:
            confidence = 0.5 * final_score / (self.threshold + 1e-10)

        # 获取模板时长
        template_obj = self._templates.get(template_name)
        template_duration_ms = template_obj.duration_ms if template_obj else 0.0

        # 获取原始采样率（如果没有提供，说明已经是 target_sample_rate）
        if sample_rate is None:
            original_sr = self.target_sample_rate
        else:
            original_sr = sample_rate

        return MatchResult(
            template_name=template_name,
            score=final_score,
            matched=matched,
            timestamp=current_time,
            position=position,
            confidence=confidence,
            sample_rate=self.target_sample_rate,
            original_sample_rate=original_sr,
            template_duration_ms=template_duration_ms,
        )

    def match_all(
        self,
        window_data: np.ndarray,
        sample_rate: int = None,
        check_cooldown: bool = True,
        use_sliding: bool = False,
        window_duration: float = 15.0,
        window_stride: float = 5.0,
    ) -> list[MatchResult]:
        """
        对所有模板进行匹配（优化版：只预处理一次窗口数据，可选滑动窗口）。

        :param window_data: 窗口音频数据
        :param sample_rate: 窗口采样率
        :param check_cooldown: 是否检查冷却时间
        :param use_sliding: 是否使用滑动窗口策略（适用于长音频）
        :param window_duration: 滑动窗口时长（秒）
        :param window_stride: 滑动窗口步长（秒）

        :return: 所有模板的匹配结果列表
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        # 如果启用滑动窗口且音频足够长，使用分段匹配
        if use_sliding:
            audio_duration = len(window_data) / sample_rate
            if audio_duration > window_duration * 1.5:
                return self.match_all_sliding(
                    window_data,
                    sample_rate,
                    check_cooldown,
                    window_duration,
                    window_stride,
                )

        # 预处理窗口数据（只做一次，包括频谱特征）
        preprocessed_window = self.features_extractor.extract(
            window_data, sample_rate, compute_centered=True, compute_spectral=True
        )

        results = []
        for template_name in self._templates:
            result = self.match_single(
                window_data,
                template_name,
                sample_rate,
                check_cooldown,
                preprocessed_window=preprocessed_window,
            )
            results.append(result)
        return results

    def match_all_sliding(
        self,
        window_data: np.ndarray,
        sample_rate: int = None,
        check_cooldown: bool = True,
        window_duration: float = 15.0,
        window_stride: float = 5.0,
    ) -> list[MatchResult]:
        """
        使用滑动窗口策略对所有模板进行匹配。

        将长音频分为多个重叠窗口，每个窗口独立匹配，返回每个模板的最佳结果。

        :param window_data: 窗口音频数据
        :param sample_rate: 窗口采样率
        :param check_cooldown: 是否检查冷却时间
        :param window_duration: 滑动窗口时长（秒）
        :param window_stride: 滑动窗口步长（秒）

        :return: 所有模板的最佳匹配结果列表
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        # 计算窗口参数
        window_samples = int(window_duration * sample_rate)
        stride_samples = int(window_stride * sample_rate)
        audio_length = len(window_data)

        # 存储每个模板的所有分段匹配结果
        template_results = {name: [] for name in self._templates}

        # 滑动窗口遍历
        for start_pos in range(0, audio_length - window_samples + 1, stride_samples):
            end_pos = start_pos + window_samples
            segment = window_data[start_pos:end_pos]

            # 对该分段执行所有模板匹配
            segment_results = self.match_all(
                segment, sample_rate, check_cooldown=False, use_sliding=False
            )

            # 调整位置偏移并存储
            for result in segment_results:
                result.position += start_pos
                template_results[result.template_name].append(result)

        # 对每个模板，选择分数最高的结果
        best_results = []
        for template_name, results_list in template_results.items():
            if results_list:
                best = max(results_list, key=lambda r: r.score)
                best_results.append(best)

        return best_results

    def match_best(
        self,
        window_data: np.ndarray,
        sample_rate: int = None,
        check_cooldown: bool = True,
    ) -> Optional[MatchResult]:
        """
        找到最佳匹配的模板。

        :param window_data: 窗口音频数据
        :param sample_rate: 窗口采样率
        :param check_cooldown: 是否检查冷却时间

        :return: 最佳匹配结果，无匹配则返回 None
        """
        results = self.match_all(window_data, sample_rate, check_cooldown)
        if not results:
            return None

        # 按分数排序，返回最高分
        best = max(results, key=lambda r: r.score)
        if best.matched:
            return best
        return None

    def detect_all_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int = None,
        template_name: str = None,
        stride_ratio: float = 0.25,
        nms_iou_threshold: float = 0.3,
    ) -> list[MatchResult]:
        """
        检测音频中所有匹配的片段（支持多次出现）。

        使用滑动窗口扫描整个音频，检测所有超过阈值的匹配片段，
        并使用非极大值抑制（NMS）去除重叠的低分片段。

        :param audio_data: 完整音频数据
        :param sample_rate: 音频采样率
        :param template_name: 指定模板名称，None则对所有模板检测
        :param stride_ratio: 滑动步长相对于模板长度的比例（0.25表示25%重叠）
        :param nms_iou_threshold: NMS的IoU阈值，超过此值的重叠片段会被抑制

        :return: 所有检测到的匹配片段列表（已按时间排序）
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        # 预处理音频数据：转换为单声道并重采样到 target_sample_rate
        original_sr = sample_rate
        preprocessed_audio = self.features_extractor.preprocess(
            audio_data, sample_rate, to_mono=True, resample=True
        )

        # 确定要检测的模板列表
        if template_name:
            if template_name not in self._templates:
                return []
            template_names = [template_name]
        else:
            template_names = list(self._templates.keys())

        all_detections = []

        # 对每个模板进行检测
        for tmpl_name in template_names:
            template = self._templates[tmpl_name]
            template_features = self._template_features[tmpl_name]

            # 计算模板长度（采样点数）
            template_length = len(template_features.data)

            # 计算滑动步长
            stride = max(1, int(template_length * stride_ratio))

            # 滑动窗口扫描
            audio_length = len(preprocessed_audio)

            for start_pos in range(0, audio_length - template_length + 1, stride):
                end_pos = start_pos + template_length
                window = preprocessed_audio[start_pos:end_pos]

                # 对窗口进行匹配
                result = self.match_single(
                    window,
                    tmpl_name,
                    sample_rate=self.target_sample_rate,  # 使用重采样后的采样率
                    check_cooldown=False,
                )

                # 只保留匹配成功的结果
                if result.matched and result.score >= self.threshold:
                    # 调整position为音频中的绝对位置
                    result.position = start_pos
                    result.sample_rate = self.target_sample_rate
                    result.original_sample_rate = original_sr
                    all_detections.append(result)

        # 如果没有检测到任何片段，直接返回
        if not all_detections:
            return []

        # 按模板和分数分组，对每个模板的检测结果应用NMS
        template_groups = {}
        for det in all_detections:
            if det.template_name not in template_groups:
                template_groups[det.template_name] = []
            template_groups[det.template_name].append(det)

        # 对每个模板应用NMS
        final_detections = []
        max_per_template = 5
        for tmpl_name, detections in template_groups.items():
            # 按分数降序排序
            detections.sort(key=lambda x: x.score, reverse=True)

            # 应用NMS
            keep = []
            while detections:
                # 取出最高分的检测
                best = detections.pop(0)
                keep.append(best)

                # 计算最佳检测的时间范围
                best_start = best.position / sample_rate
                best_end = best_start + (best.template_duration_ms / 1000.0)

                # 过滤掉与最佳检测重叠过多的其他检测
                remaining = []
                for det in detections:
                    det_start = det.position / sample_rate
                    det_end = det_start + (det.template_duration_ms / 1000.0)

                    # 计算IoU
                    intersection_start = max(best_start, det_start)
                    intersection_end = min(best_end, det_end)
                    intersection = max(0, intersection_end - intersection_start)

                    union_start = min(best_start, det_start)
                    union_end = max(best_end, det_end)
                    union = union_end - union_start

                    iou = intersection / union if union > 0 else 0

                    # 如果IoU小于阈值，保留该检测
                    if iou < nms_iou_threshold:
                        remaining.append(det)

                detections = remaining

            # 仅保留前 max_per_template 个结果
            final_detections.extend(keep[:max_per_template])

        # 按时间排序
        final_detections.sort(key=lambda x: x.position)

        return final_detections

    @staticmethod
    def _pick_peaks_1d(
        scores: np.ndarray,
        min_height: float,
        min_distance: int,
    ) -> np.ndarray:
        """Pick local maxima indices from a 1D score array.

        A peak is an index i such that scores[i-1] < scores[i] >= scores[i+1] and
        scores[i] >= min_height. Peaks closer than min_distance keep only the
        highest one.
        """
        if scores is None or scores.size < 3:
            return np.array([], dtype=np.int64)

        # Local maxima mask (strict on left, non-strict on right to keep plateau starts)
        mid = scores[1:-1]
        peaks_mask = (scores[:-2] < mid) & (mid >= scores[2:]) & (mid >= min_height)
        peak_idxs = np.nonzero(peaks_mask)[0] + 1

        if peak_idxs.size == 0:
            return peak_idxs.astype(np.int64)

        if min_distance <= 1:
            # Just sort by index (time)
            return np.sort(peak_idxs).astype(np.int64)

        # Enforce min_distance by greedy selection on score (highest first)
        order = np.argsort(scores[peak_idxs])[::-1]
        selected: list[int] = []
        for k in order:
            idx = int(peak_idxs[k])
            if all(abs(idx - s) >= min_distance for s in selected):
                selected.append(idx)

        return np.array(sorted(selected), dtype=np.int64)

    @staticmethod
    def _merge_nearby_segments(
        segments: list[MatchResult],
        merge_gap_seconds: float,
        sample_rate: int,
    ) -> list[MatchResult]:
        """Merge segments that are very close in time (same template)."""
        if not segments:
            return []

        segments = sorted(
            segments, key=lambda r: (r.template_name, r.position, -r.score)
        )
        merged: list[MatchResult] = []

        current = segments[0]
        for seg in segments[1:]:
            if seg.template_name != current.template_name:
                merged.append(current)
                current = seg
                continue

            cur_end = current.position / sample_rate + (
                current.template_duration_ms / 1000.0
            )
            seg_start = seg.position / sample_rate
            if seg_start - cur_end <= merge_gap_seconds:
                # Merge duplicates by keeping the higher-score one
                if seg.score > current.score:
                    current = seg
            else:
                merged.append(current)
                current = seg

        merged.append(current)
        return merged

    def detect_all_segments_v2(
        self,
        audio_data: np.ndarray,
        sample_rate: int = None,
        template_name: str = None,
        peak_threshold: float | None = None,
        peak_min_distance_ratio: float = 0.75,
        merge_gap_seconds: float = 0.10,
        max_per_template: int = 8,
    ) -> list[MatchResult]:
        """Detect all matching segments using full correlation sequence + peak picking.

        Compared to `detect_all_segments`, this avoids scanning with many windows.
        It computes one NCC-like score sequence per template over the entire audio
        and then picks peaks as candidate occurrences.
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        original_sr = sample_rate
        preprocessed_audio = self.features_extractor.preprocess(
            audio_data, sample_rate, to_mono=True, resample=True
        )
        sr = self.target_sample_rate

        if template_name:
            if template_name not in self._templates:
                return []
            template_names = [template_name]
        else:
            template_names = list(self._templates.keys())

        detections: list[MatchResult] = []

        for tmpl_name in template_names:
            template_features = self._template_features[tmpl_name]
            template_zm = template_features.data
            template_len = len(template_zm)
            if template_len < 8 or len(preprocessed_audio) < template_len:
                continue

            template_energy = float(np.sum(template_zm**2))
            if template_energy <= 0:
                continue

            window_len = len(preprocessed_audio)

            fft_len = 1
            while fft_len < window_len + template_len - 1:
                fft_len *= 2

            template_zm_fft = rfft(template_zm[::-1], fft_len)
            audio_fft = rfft(preprocessed_audio, fft_len)
            correlation = irfft(audio_fft * template_zm_fft, fft_len)

            valid_start = template_len - 1
            valid_len = window_len - template_len + 1
            correlation = correlation[valid_start : valid_start + valid_len]

            audio_sq = preprocessed_audio**2
            cumsum_sq = np.concatenate(([0.0], np.cumsum(audio_sq)))
            window_sq_sum = cumsum_sq[template_len:] - cumsum_sq[:-template_len]

            cumsum_val = np.concatenate(([0.0], np.cumsum(preprocessed_audio)))
            window_sum = cumsum_val[template_len:] - cumsum_val[:-template_len]
            window_mean = window_sum / template_len
            window_std_energy = window_sq_sum - template_len * window_mean**2
            window_std_energy = np.maximum(window_std_energy, 0.0)

            denominator = np.sqrt(window_std_energy * template_energy)

            min_energy_ratio = 0.15
            valid_mask = window_std_energy >= (min_energy_ratio * template_energy)
            safe_den = np.where(denominator > 0, denominator, 1.0)
            scores = np.where(valid_mask, correlation / safe_den, -np.inf)

            # Clamp to [0, 1] and drop non-finite values
            scores = np.where(np.isfinite(scores), np.clip(scores, 0.0, 1.0), 0.0)

            thr = self.threshold if peak_threshold is None else float(peak_threshold)
            min_distance = max(1, int(template_len * peak_min_distance_ratio))
            peak_idxs = self._pick_peaks_1d(
                scores, min_height=thr, min_distance=min_distance
            )
            if peak_idxs.size == 0:
                continue

            # Keep strongest peaks first (then cap per template)
            peak_order = np.argsort(scores[peak_idxs])[::-1]
            peak_idxs = peak_idxs[peak_order][:max_per_template]
            peak_idxs = np.sort(peak_idxs)

            template_obj = self._templates.get(tmpl_name)
            template_duration_ms = template_obj.duration_ms if template_obj else 0.0

            now = time.time()
            for pos in peak_idxs:
                s = float(scores[int(pos)])
                detections.append(
                    MatchResult(
                        template_name=tmpl_name,
                        score=s,
                        matched=True,
                        timestamp=now,
                        position=int(pos),
                        confidence=min(
                            max((s - thr) / max(1e-10, (1.0 - thr)), 0.0), 1.0
                        ),
                        sample_rate=sr,
                        original_sample_rate=original_sr,
                        template_duration_ms=template_duration_ms,
                    )
                )

        if not detections:
            return []

        # Merge very close duplicates per template then sort by time
        detections = self._merge_nearby_segments(detections, merge_gap_seconds, sr)
        detections.sort(key=lambda r: r.position)
        return detections

    def _compute_envelope_ncc_sequence(
        self,
        audio_data: np.ndarray,
        template_features: "AudioFeatures",
    ) -> np.ndarray:
        """Compute NCC score sequence between audio envelope and template envelope.

        The sequence index is in envelope-frame units (hop_length samples per frame).
        """
        # Compute audio envelope in hop-length frames
        hop = self.features_extractor.hop_length
        audio_env = self.features_extractor.compute_energy_envelope(
            audio_data, frame_length=hop
        )
        template_env = template_features.energy_envelope
        if audio_env is None or template_env is None:
            return np.array([], dtype=np.float32)

        if len(audio_env) < len(template_env) or len(template_env) < 3:
            return np.array([], dtype=np.float32)

        # Zero-mean template envelope for NCC
        t = template_env.astype(np.float32)
        t = t - float(np.mean(t))
        t_energy = float(np.sum(t**2))
        if t_energy <= 0:
            return np.array([], dtype=np.float32)

        x = audio_env.astype(np.float32)
        n = len(x)
        m = len(t)

        fft_len = 1
        while fft_len < n + m - 1:
            fft_len *= 2

        t_fft = rfft(t[::-1], fft_len)
        x_fft = rfft(x, fft_len)
        corr = irfft(x_fft * t_fft, fft_len)

        valid_start = m - 1
        valid_len = n - m + 1
        corr = corr[valid_start : valid_start + valid_len]

        # sliding window normalization for x
        x_sq = x**2
        cumsum_sq = np.concatenate(([0.0], np.cumsum(x_sq)))
        x_sq_sum = cumsum_sq[m:] - cumsum_sq[:-m]

        cumsum_val = np.concatenate(([0.0], np.cumsum(x)))
        x_sum = cumsum_val[m:] - cumsum_val[:-m]
        x_mean = x_sum / m
        x_var_energy = x_sq_sum - m * x_mean**2
        x_var_energy = np.maximum(x_var_energy, 0.0)

        denom = np.sqrt(x_var_energy * t_energy)
        safe = np.where(denom > 0, denom, 1.0)

        # small energy gating to avoid silence peaks
        min_energy_ratio = 0.10
        valid_mask = x_var_energy >= (min_energy_ratio * t_energy)
        scores = np.where(valid_mask, corr / safe, -np.inf)
        scores = np.where(np.isfinite(scores), np.clip(scores, 0.0, 1.0), 0.0)
        return scores.astype(np.float32)

    def calibrate_thresholds_from_negatives(
        self,
        sounds_dir: Path | None = None,
        quantile: float = 0.999,
        margin: float = 0.02,
        min_threshold: float = 0.70,
        max_threshold: float = 0.95,
    ) -> dict[str, float]:
        """Calibrate per-template coarse thresholds using negative files in refs.json.

        Uses envelope NCC sequence maxima on each negative file.
        Returns a dict {template_name: threshold}.
        """
        if sounds_dir is None:
            sounds_dir = TEST_SOUNDS_DIR
        sounds_dir = Path(sounds_dir)

        refs_path = sounds_dir / "refs.json"
        if not refs_path.exists():
            return {}

        refs_data = json.loads(refs_path.read_text(encoding="utf-8"))
        negative_files = [
            k
            for k, v in refs_data.items()
            if float(v.get("start_time", -1)) < 0 or float(v.get("end_time", -1)) < 0
        ]
        if not negative_files:
            return {}

        # resolve file paths
        neg_paths: list[Path] = []
        for fname in negative_files:
            for p in sounds_dir.rglob(fname):
                neg_paths.append(p)
                break

        if not neg_paths:
            return {}

        per_template_maxes: dict[str, list[float]] = {
            name: [] for name in self._templates
        }

        for wav_path in neg_paths:
            try:
                audio, sr0 = sf.read(wav_path)
            except Exception:
                continue

            # match test behavior: skip first 5 seconds
            skip_s = 5.0
            skip_n = int(skip_s * sr0)
            if len(audio) <= skip_n:
                continue
            audio = audio[skip_n:]

            # preprocess to target SR
            audio = self.features_extractor.preprocess(
                audio, sr0, to_mono=True, resample=True
            )

            for tmpl_name in self._templates:
                template_features = self._template_features[tmpl_name]
                seq = self._compute_envelope_ncc_sequence(audio, template_features)
                if seq.size == 0:
                    per_template_maxes[tmpl_name].append(0.0)
                else:
                    per_template_maxes[tmpl_name].append(float(np.max(seq)))

        thresholds: dict[str, float] = {}
        for tmpl_name, vals in per_template_maxes.items():
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size == 0:
                continue
            # With very few negative files, quantiles can be misleading; use a conservative base.
            # Base is max(quantile, max) so any observed negative max is covered.
            base_q = float(np.quantile(arr, quantile))
            base_m = float(np.max(arr))
            base = max(base_q, base_m)
            thr = base + float(margin)
            thr = float(np.clip(thr, min_threshold, max_threshold))
            thresholds[tmpl_name] = thr

        return thresholds

    def detect_all_segments_v3(
        self,
        audio_data: np.ndarray,
        sample_rate: int = None,
        template_name: str = None,
        coarse_thresholds: dict[str, float] | None = None,
        peak_min_distance_ratio: float = 0.60,
        refine_search_seconds: float = 0.35,
        refine_accept_threshold: float | None = None,
        merge_gap_seconds: float = 0.10,
        max_per_template: int = 8,
    ) -> list[MatchResult]:
        """Two-stage detector: envelope coarse peaks -> match_single refinement."""
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        original_sr = sample_rate
        audio = self.features_extractor.preprocess(
            audio_data, sample_rate, to_mono=True, resample=True
        )
        sr = self.target_sample_rate
        hop = self.features_extractor.hop_length

        if template_name:
            if template_name not in self._templates:
                return []
            template_names = [template_name]
        else:
            template_names = list(self._templates.keys())

        coarse_thresholds = coarse_thresholds or {}
        if refine_accept_threshold is None:
            # Stricter than global threshold to suppress false positives
            refine_accept_threshold = min(0.95, max(self.threshold + 0.03, 0.86))

        detections: list[MatchResult] = []
        now = time.time()

        for tmpl_name in template_names:
            template_features = self._template_features[tmpl_name]
            template_len = len(template_features.data)
            if template_len < 8 or len(audio) < template_len:
                continue

            seq = self._compute_envelope_ncc_sequence(audio, template_features)
            if seq.size == 0:
                continue

            thr = float(
                coarse_thresholds.get(tmpl_name, max(0.72, self.threshold - 0.05))
            )
            min_distance_frames = max(
                1, int((template_len / hop) * peak_min_distance_ratio)
            )
            peak_frames = self._pick_peaks_1d(
                seq.astype(np.float64), min_height=thr, min_distance=min_distance_frames
            )
            if peak_frames.size == 0:
                continue

            # sort by coarse score desc and cap
            order = np.argsort(seq[peak_frames])[::-1]
            peak_frames = peak_frames[order][:max_per_template]

            template_obj = self._templates.get(tmpl_name)
            template_duration_ms = template_obj.duration_ms if template_obj else 0.0

            # refine around each coarse peak
            refine_half = int(refine_search_seconds * sr)
            refined_candidates: list[MatchResult] = []
            for pf in peak_frames:
                approx_pos = int(pf) * hop
                start = max(0, approx_pos - refine_half)
                end = min(len(audio), approx_pos + template_len + refine_half)
                region = audio[start:end]
                if len(region) < template_len:
                    continue

                refined = self.match_single(
                    region,
                    tmpl_name,
                    sample_rate=sr,
                    check_cooldown=False,
                )
                if (not refined.matched) or (
                    refined.score < float(refine_accept_threshold)
                ):
                    continue

                refined.position = start + refined.position
                refined.sample_rate = sr
                refined.original_sample_rate = original_sr
                refined.timestamp = now
                refined.template_duration_ms = template_duration_ms
                refined_candidates.append(refined)

            # Per-template: keep only top-K refined detections
            if refined_candidates:
                refined_candidates.sort(key=lambda r: r.score, reverse=True)
                detections.extend(refined_candidates[:max_per_template])

        if not detections:
            return []

        detections = self._merge_nearby_segments(detections, merge_gap_seconds, sr)
        detections.sort(key=lambda r: r.position)
        return detections

    def reset_cooldowns(self):
        """重置所有模板的冷却时间。"""
        for name in self._last_match_times:
            self._last_match_times[name] = 0

    @property
    def template_count(self) -> int:
        """获取模板数量。"""
        return len(self._templates)

    def __repr__(self) -> str:
        return (
            f"TemplateMatcher("
            f"templates={self.template_count}, "
            f"threshold={self.threshold}, "
            f"sample_rate={self.target_sample_rate}, "
            f"cooldown_ms={self.cooldown_ms}, "
            f"weights=[time:{self.weight_time_corr:.2f}, "
            f"mel:{self.weight_mel_spec:.2f}, "
            f"env:{self.weight_energy_env:.2f}, "
            f"zcr:{self.weight_zcr:.2f}, "
            f"centroid:{self.weight_spectral_centroid:.2f}])"
        )


class RealtimeMatcher:
    """
    实时模板匹配器。

    将 SoundRecorder 和 TemplateMatcher 结合，实现实时音频模板匹配。
    """

    def __init__(
        self,
        recorder: SoundRecorder = None,
        matcher: TemplateMatcher = None,
        match_callback: Callable[[MatchResult], Any] = None,
        match_interval_ms: float = 100,
    ):
        """
        初始化实时匹配器。

        :param recorder: 音频录制器，默认自动创建
        :param matcher: 模板匹配器，默认自动创建
        :param match_callback: 匹配成功时的回调函数
        :param match_interval_ms: 匹配间隔（毫秒）
        """
        self.recorder = recorder or SoundRecorder()
        self.matcher = matcher or TemplateMatcher()
        self.match_callback = match_callback
        self.match_interval_ms = match_interval_ms

        # 匹配状态
        self._is_running = False
        self._match_count = 0
        self._match_thread: Optional[threading.Thread] = None

    def add_template(self, template: AudioTemplate):
        """添加模板。"""
        self.matcher.add_template(template)

    def load_templates(self, dir_path: Path = None) -> int:
        """
        从目录加载模板。

        :param dir_path: 模板目录

        :return: 加载的模板数量
        """
        loader = TemplateLoader()
        count = loader.load_directory(dir_path)
        for template in loader.get_all_templates():
            self.matcher.add_template(template)
        return count

    def _match_loop(self):
        """匹配循环（在独立线程中运行）。"""
        while self._is_running:
            # 获取窗口数据
            window_data = self.recorder.buffer.get_window_data()
            if window_data is not None and len(window_data) > 0:
                # 执行匹配
                result = self.matcher.match_best(
                    window_data,
                    sample_rate=self.recorder.sample_rate,
                )
                if result and result.matched:
                    self._match_count += 1
                    self._log_match_result(result)
                    if self.match_callback:
                        try:
                            self.match_callback(result)
                        except Exception as e:
                            logger.warn(f"匹配回调执行出错: {e}")

            time.sleep(self.match_interval_ms / 1000)

    def _log_match_result(self, result: MatchResult):
        """输出匹配结果日志。"""
        print()
        logger.note("=" * 50)
        logger.okay(f"检测到音频模板 #{self._match_count}: {result.template_name}")
        logger.note("=" * 50)
        time_str = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
        ms = int((result.timestamp % 1) * 1000)
        info_dict = {
            "触发时间": f"{time_str}.{ms:03d}",
            "匹配分数": f"{result.score:.3f}",
            "置信度": f"{result.confidence:.3f}",
        }
        logger.note(dict_to_lines(info_dict, key_prefix="* "))

    def start(self):
        """启动实时匹配。"""
        if self._is_running:
            logger.warn("实时匹配器已在运行中")
            return

        # 启动录制器的音频流
        if not self.recorder.is_streaming:
            if not self.recorder.start_stream():
                logger.warn("无法启动音频流")
                return

        # 启动匹配线程
        self._is_running = True
        self._match_thread = threading.Thread(target=self._match_loop, daemon=True)
        self._match_thread.start()

        logger.okay("实时匹配器已启动")
        info_dict = {
            "模板数量": self.matcher.template_count,
            "匹配阈值": self.matcher.threshold,
            "匹配间隔": f"{self.match_interval_ms}ms",
            "冷却时间": f"{self.matcher.cooldown_ms}ms",
        }
        logger.note(dict_to_lines(info_dict, key_prefix="* "))

    def stop(self):
        """停止实时匹配。"""
        if not self._is_running:
            logger.warn("实时匹配器未在运行")
            return

        self._is_running = False
        if self._match_thread:
            self._match_thread.join(timeout=1.0)
            self._match_thread = None

        logger.okay(f"实时匹配器已停止，共检测到 {self._match_count} 次匹配")

    def run(self, duration: float = None):
        """
        运行实时匹配。

        :param duration: 运行时长（秒），None 表示持续运行直到 Ctrl+C
        """
        self.start()

        try:
            if duration:
                logger.note(f"定时模式：{duration} 秒...")
                start_time = time.time()
                while time.time() - start_time < duration:
                    time.sleep(0.1)
            else:
                logger.note(f"持续模式：按 {key_hint('Ctrl+C')} 停止...")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.note(f"\n检测到 {key_hint('Ctrl+C')}，正在退出...")
        finally:
            self.stop()
            self.recorder.stop_stream()

    @property
    def is_running(self) -> bool:
        """获取运行状态。"""
        return self._is_running

    @property
    def match_count(self) -> int:
        """获取匹配次数。"""
        return self._match_count

    def __repr__(self) -> str:
        return (
            f"RealtimeMatcher("
            f"templates={self.matcher.template_count}, "
            f"threshold={self.matcher.threshold}, "
            f"is_running={self.is_running}, "
            f"match_count={self._match_count})"
        )


def compute_iou(
    pred_start: float, pred_end: float, ref_start: float, ref_end: float
) -> float:
    """
    计算预测时间段和参考时间段的 IoU (Intersection over Union)。

    :param pred_start: 预测开始时间（秒）
    :param pred_end: 预测结束时间（秒）
    :param ref_start: 参考开始时间（秒）
    :param ref_end: 参考结束时间（秒）

    :return: IoU 值 (0-1)
    """
    # 计算交集
    intersection_start = max(pred_start, ref_start)
    intersection_end = min(pred_end, ref_end)
    intersection = max(0, intersection_end - intersection_start)

    # 计算并集
    union_start = min(pred_start, ref_start)
    union_end = max(pred_end, ref_end)
    union = union_end - union_start

    if union <= 0:
        return 0.0

    return intersection / union


def compute_offset(pred_time: float, ref_time: float) -> float:
    """
    计算预测时间和参考时间的偏移（秒）。

    :param pred_time: 预测时间（秒）
    :param ref_time: 参考时间（秒）

    :return: 偏移值（秒），正值表示预测偏晚，负值表示预测偏早
    """
    return pred_time - ref_time


def test_templates(
    templates_dir: Path = None,
    sounds_dir: Path = None,
    threshold: float = MATCH_THRESHOLD,
    prefix: str = TEMPLATE_PREFIX,
) -> dict[str, list[MatchResult]]:
    """
    测试模板匹配效果。

    从测试音频目录加载音频文件，对每个文件执行模板匹配并输出结果。

    :param templates_dir: 模板目录，默认使用 TEMPLATES_DIR
    :param sounds_dir: 测试音频目录，默认使用 TEST_SOUNDS_DIR
    :param threshold: 匹配阈值
    :param prefix: 模板文件前缀过滤

    :return: 测试结果字典，键为测试文件路径，值为匹配结果列表
    """
    if templates_dir is None:
        templates_dir = TEMPLATES_DIR
    if sounds_dir is None:
        sounds_dir = TEST_SOUNDS_DIR

    templates_dir = Path(templates_dir)
    sounds_dir = Path(sounds_dir)

    # 加载参考时间数据
    refs_path = sounds_dir / "refs.json"
    refs_data = {}
    if refs_path.exists():
        with open(refs_path, "r", encoding="utf-8") as f:
            refs_data = json.load(f)
        logger.note(f"已加载参考时间数据: {refs_path}")
    else:
        logger.warn(f"未找到参考时间文件: {refs_path}")

    # 加载模板
    logger.note(f"正在从 {templates_dir} 加载模板...")
    loader = TemplateLoader(templates_dir)
    template_count = loader.load_directory(prefix=prefix)

    if template_count == 0:
        logger.warn("没有找到任何模板文件")
        return {}

    # 创建匹配器
    matcher = TemplateMatcher(
        templates=loader.get_all_templates(),
        threshold=threshold,
    )

    # 使用负例自校准粗检阈值（用于 v3 二阶段检测）
    calibrated_thresholds = matcher.calibrate_thresholds_from_negatives(
        sounds_dir=sounds_dir,
        quantile=0.9995,
        margin=0.02,
        min_threshold=0.70,
        max_threshold=0.95,
    )
    if calibrated_thresholds:
        logger.note(
            "已基于负例自校准粗检阈值: "
            + ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(calibrated_thresholds.items())
            )
        )

    # 收集测试音频文件
    test_files: list[Path] = []
    extensions = [".wav", ".flac", ".ogg", ".mp3"]

    if not sounds_dir.exists():
        logger.warn(f"测试音频目录不存在: {sounds_dir}")
        return {}

    # 遍历子目录和文件
    for item in sorted(sounds_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in extensions:
            test_files.append(item)
        elif item.is_dir():
            for file in sorted(item.iterdir()):
                if file.is_file() and file.suffix.lower() in extensions:
                    test_files.append(file)

    if not test_files:
        logger.warn(f"在 {sounds_dir} 中没有找到测试音频文件")
        return {}

    logger.note(f"找到 {len(test_files)} 个测试音频文件")
    logger.note("=" * 60)

    # 执行测试
    results: dict[str, list[MatchResult]] = {}
    all_results_data = {}  # 合并所有结果
    total_match_time = 0.0

    # 准确率统计（最佳匹配）
    total_with_ref = 0  # 有参考数据的测试数
    accurate_matches = 0  # IoU > 0.5 的匹配数（最佳匹配）
    total_best_iou = 0.0
    total_best_start_offset = 0.0
    total_best_end_offset = 0.0

    # 全模板统计
    total_all_iou = 0.0  # 所有模板的总IoU
    total_all_start_offset = 0.0
    total_all_end_offset = 0.0
    total_all_count = 0  # 所有模板匹配的总数
    worst_iou = 1.0  # 最差IoU
    worst_iou_info = ""  # 最差IoU的信息

    # 多片段评估统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_fp_near_ref = 0
    total_fp_far = 0
    negative_files = 0
    negative_fp_files = 0
    negative_fp_segments = 0

    for test_file in test_files:
        # 读取测试音频
        try:
            data, sample_rate = sf.read(test_file)
        except Exception as e:
            logger.warn(f"读取测试文件失败 {test_file}: {e}")
            continue

        # 跳过前 5 秒的音频（目标匹配通常在 5 秒之后）
        skip_seconds = 5.0
        skip_samples = int(skip_seconds * sample_rate)
        if len(data) > skip_samples:
            data = data[skip_samples:]
        else:
            logger.warn(f"音频文件太短，跳过: {test_file}")
            continue

        # 执行多片段检测（禁用冷却）并计时
        start_time = time.perf_counter()
        all_segments = matcher.detect_all_segments_v3(
            data,
            sample_rate=sample_rate,
            template_name=None,  # 检测所有模板
            coarse_thresholds=calibrated_thresholds,
            peak_min_distance_ratio=0.85,
            refine_search_seconds=0.45,
            refine_accept_threshold=threshold,
            merge_gap_seconds=0.10,
            max_per_template=3,
        )
        match_time = (time.perf_counter() - start_time) * 1000
        total_match_time += match_time

        # 将匹配位置补偿回原始音频位置（加上跳过的 5 秒）
        for result in all_segments:
            # position 以 result.sample_rate 为单位（target_sample_rate），直接加上跳过段的采样点
            result.position += int(skip_seconds * result.sample_rate)

        # 记录结果
        rel_path = test_file.relative_to(sounds_dir)
        results[str(rel_path)] = all_segments

        # 按分数排序
        sorted_results = sorted(all_segments, key=lambda r: r.score, reverse=True)
        best = sorted_results[0] if sorted_results else None

        # 获取参考时间
        file_name = test_file.name
        ref_info = refs_data.get(file_name, {})
        ref_start = ref_info.get("start_time", -1)
        ref_end = ref_info.get("end_time", -1)
        has_ref = ref_start >= 0 and ref_end >= 0

        # 计算最佳匹配的 IoU 和偏移度
        best_iou = 0.0
        best_start_offset = 0.0
        best_end_offset = 0.0
        is_accurate = False

        # 计算所有模板的指标
        all_ious = []
        all_start_offsets = []
        all_end_offsets = []
        all_template_metrics = []

        if has_ref:
            # 计算所有模板的IoU
            for result in sorted_results:
                result_dict = result.to_dict()
                pred_start = float(result_dict["start_time"])
                pred_end = float(result_dict["end_time"])

                r_iou = compute_iou(pred_start, pred_end, ref_start, ref_end)
                r_start_offset = compute_offset(pred_start, ref_start)
                r_end_offset = compute_offset(pred_end, ref_end)

                all_ious.append(r_iou)
                all_start_offsets.append(r_start_offset)
                all_end_offsets.append(r_end_offset)

                all_template_metrics.append(
                    {
                        "template_name": result.template_name,
                        "score": round(result.score, 4),
                        "iou": round(r_iou, 4),
                        "start_offset": round(r_start_offset, 4),
                        "end_offset": round(r_end_offset, 4),
                    }
                )

                # 更新全局最差IoU
                if r_iou < worst_iou:
                    worst_iou = r_iou
                    worst_iou_info = f"{file_name} - {result.template_name}"

            # 最佳匹配的指标
            if best:
                best_iou = all_ious[0]
                best_start_offset = all_start_offsets[0]
                best_end_offset = all_end_offsets[0]
                is_accurate = best_iou > 0.5

                total_with_ref += 1
                total_best_iou += best_iou
                total_best_start_offset += abs(best_start_offset)
                total_best_end_offset += abs(best_end_offset)

                if is_accurate:
                    accurate_matches += 1

                # 累计所有模板的指标
                total_all_iou += sum(all_ious)
                total_all_start_offset += sum(abs(o) for o in all_start_offsets)
                total_all_end_offset += sum(abs(o) for o in all_end_offsets)
                total_all_count += len(all_ious)

            # 多片段评估：基于 IoU 的 one-to-one 匹配
            pred_intervals = []
            for result in sorted_results:
                d = result.to_dict()
                pred_intervals.append((float(d["start_time"]), float(d["end_time"])))

            ref_interval = (float(ref_start), float(ref_end))
            ious_for_pred = [
                compute_iou(ps, pe, ref_interval[0], ref_interval[1])
                for (ps, pe) in pred_intervals
            ]

            # 视为“有命中”阈值
            hit_iou_threshold = 0.5
            has_hit = any(i >= hit_iou_threshold for i in ious_for_pred)

            # 额外：统计 FP 的“距离参考段远近”
            # 认为与参考段中心点相距 <= 3s 的检测属于 near-ref（更可能是“找到了但没对齐/泛化失败”）
            ref_center = (float(ref_start) + float(ref_end)) / 2.0
            near_ref_window_s = 3.0
            fp_near = 0
            fp_far = 0
            for (ps, pe), iou_val in zip(pred_intervals, ious_for_pred):
                pred_center = (ps + pe) / 2.0
                if iou_val >= hit_iou_threshold:
                    continue
                if abs(pred_center - ref_center) <= near_ref_window_s:
                    fp_near += 1
                else:
                    fp_far += 1
            total_fp_near_ref += fp_near
            total_fp_far += fp_far

            if has_hit:
                total_tp += 1
                # 其他预测都算 FP（因为只有一个参考段）
                total_fp += max(0, len(pred_intervals) - 1)
                total_fn += 0
            else:
                total_tp += 0
                total_fp += len(pred_intervals)
                total_fn += 1
        else:
            # 负例：所有检测均为 FP
            negative_files += 1
            if sorted_results:
                negative_fp_files += 1
                negative_fp_segments += len(sorted_results)
            total_fp += len(sorted_results)

        # 保存结果到同名 JSON 文件（添加 _match 后缀）
        json_path = test_file.with_name(test_file.stem + "_match.json")
        json_data = {
            "file": test_file.name,
            "match_time_ms": round(match_time, 2),
            "best_match": best.to_dict() if best else None,
            "all_matches": [r.to_dict() for r in sorted_results],
            "reference": (
                {
                    "start_time": ref_start,
                    "end_time": ref_end,
                }
                if has_ref
                else None
            ),
            "metrics": (
                {
                    "best_iou": round(best_iou, 4),
                    "best_start_offset": round(best_start_offset, 4),
                    "best_end_offset": round(best_end_offset, 4),
                    "is_accurate": is_accurate,
                    "avg_iou": (
                        round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0
                    ),
                    "worst_iou": round(min(all_ious), 4) if all_ious else 0.0,
                    "avg_start_offset": (
                        round(
                            sum(abs(o) for o in all_start_offsets)
                            / len(all_start_offsets),
                            4,
                        )
                        if all_start_offsets
                        else 0.0
                    ),
                    "avg_end_offset": (
                        round(
                            sum(abs(o) for o in all_end_offsets) / len(all_end_offsets),
                            4,
                        )
                        if all_end_offsets
                        else 0.0
                    ),
                }
                if has_ref
                else None
            ),
            "all_template_metrics": all_template_metrics if has_ref else None,
            "segment_metrics": (
                {
                    "is_negative": (not has_ref),
                    "segment_count": len(sorted_results),
                    "has_hit_iou_0_5": (
                        None
                        if not has_ref
                        else any(
                            compute_iou(
                                float(r.to_dict()["start_time"]),
                                float(r.to_dict()["end_time"]),
                                float(ref_start),
                                float(ref_end),
                            )
                            >= 0.5
                            for r in sorted_results
                        )
                    ),
                    "fp_near_ref": (None if not has_ref else fp_near),
                    "fp_far": (None if not has_ref else fp_far),
                }
            ),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 添加到合并结果
        all_results_data[file_name] = json_data

        # 输出结果
        logger.file(f"\n测试文件: {rel_path}")

        # 判断是否为负例
        is_negative = not has_ref

        if not sorted_results:
            # 没有检测到任何片段
            if is_negative:
                logger.okay(f"  ✓ 负例：未检测到任何匹配片段（正确）")
            else:
                logger.warn(f"  ⚠ 未检测到任何匹配片段")
        else:
            # 显示检测到的片段数量
            segment_count = len(sorted_results)
            if is_negative:
                logger.warn(f"  ⚠ 负例：检测到 {segment_count} 个片段（假阳性）")
            else:
                logger.note(f"  → 检测到 {segment_count} 个匹配片段")

            # 输出匹配结果（低分用 warn，高分用 okay）
            LOW_SCORE_THRESHOLD = 0.75
            for i, result in enumerate(sorted_results[:10], 1):  # 最多显示前10个
                result_dict = result.to_dict()
                time_info = f"[{result_dict['start_time']} ~ {result_dict['end_time']}] ({result_dict['duration']}s)"

                if result.score < LOW_SCORE_THRESHOLD:
                    # 低分结果：整行用 warn
                    logger.warn(
                        f"    #{i} {result.template_name}: {result.score:.3f} {time_info}"
                    )
                elif result.matched:
                    # 高分匹配：使用颜色样式
                    template_str = logstr.file(result.template_name)
                    score_str = logstr.okay(f"{result.score:.3f}")
                    time_str = logstr.mesg(time_info)
                    logger.okay(f"    #{i} {template_str}: {score_str} {time_str}")
                else:
                    logger.mesg(
                        f"    #{i} {result.template_name}: {result.score:.3f} {time_info}"
                    )

            if segment_count > 10:
                logger.note(f"    ... 还有 {segment_count - 10} 个片段未显示")

        # 输出最佳匹配和准确率信息
        if best and best.matched:
            best_dict = best.to_dict()
            best_template = logstr.file(best.template_name)
            best_score = logstr.okay(f"{best.score:.3f}")
            best_time = logstr.mesg(
                f"[{best_dict['start_time']}s ~ {best_dict['end_time']}s]"
            )
            logger.okay(f"  → 最佳匹配: {best_template} ({best_score}) {best_time}")

            # 如果有参考数据，输出准确率指标
            if has_ref:
                ref_time = logstr.mesg(f"[{ref_start}s ~ {ref_end}s]")
                best_iou_str = (
                    logstr.okay(f"{best_iou:.3f}")
                    if best_iou > 0.5
                    else logstr.warn(f"{best_iou:.3f}")
                )
                avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
                worst_iou_val = min(all_ious) if all_ious else 0.0
                logger.note(f"  → 参考时间: {ref_time}")
                logger.note(
                    f"  → 最佳IoU: {best_iou_str}, 开始偏移: {best_start_offset:+.2f}s, 结束偏移: {best_end_offset:+.2f}s"
                )
                logger.note(f"  → 平均IoU: {avg_iou:.3f}, 最差IoU: {worst_iou_val:.3f}")

        logger.file(f"  → 结果已保存: {json_path.name}")

    # 保存合并结果
    results_json_path = sounds_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=2)
    logger.okay(f"\n所有结果已合并保存到: {results_json_path}")

    # 输出统计信息
    logger.note("\n" + "=" * 60)
    logger.note("测试统计:")

    total_files = len(results)
    matched_files = sum(
        1 for matches in results.values() if any(r.matched for r in matches)
    )

    info_dict = {
        "模板数量": template_count,
        "测试文件数": total_files,
        "匹配成功数": matched_files,
        "匹配率": (
            f"{matched_files / total_files * 100:.1f}%" if total_files > 0 else "N/A"
        ),
        "总匹配耗时": f"{total_match_time:.1f}ms",
        "平均耗时": (
            f"{total_match_time / total_files:.1f}ms" if total_files > 0 else "N/A"
        ),
    }

    # 添加准确率统计
    if total_with_ref > 0:
        # 最佳匹配统计
        avg_best_iou = total_best_iou / total_with_ref
        avg_best_start_offset = total_best_start_offset / total_with_ref
        avg_best_end_offset = total_best_end_offset / total_with_ref
        accuracy_rate = accurate_matches / total_with_ref * 100

        # 所有模板统计
        avg_all_iou = total_all_iou / total_all_count if total_all_count > 0 else 0.0
        avg_all_start_offset = (
            total_all_start_offset / total_all_count if total_all_count > 0 else 0.0
        )
        avg_all_end_offset = (
            total_all_end_offset / total_all_count if total_all_count > 0 else 0.0
        )

        info_dict.update(
            {
                "": "",  # 空行分隔
                "有参考数据": total_with_ref,
                "准确匹配数 (IoU>0.5)": accurate_matches,
                "准确率": f"{accuracy_rate:.1f}%",
                " ": "",  # 空行分隔
                "【最佳匹配】": "",
                "平均IoU": f"{avg_best_iou:.3f}",
                "平均开始偏移": f"{avg_best_start_offset:.2f}s",
                "平均结束偏移": f"{avg_best_end_offset:.2f}s",
                "  ": "",  # 空行分隔
                "【所有模板】": "",
                "总匹配数": total_all_count,
                "全局平均IoU": f"{avg_all_iou:.3f}",
                # "全局最差IoU": f"{worst_iou:.3f} ({worst_iou_info})",
                "全局平均开始偏移": f"{avg_all_start_offset:.2f}s",
                "全局平均结束偏移": f"{avg_all_end_offset:.2f}s",
            }
        )

    # 多片段评估（TP/FP/FN，precision/recall/F1）
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    info_dict.update(
        {
            "   ": "",
            "【多片段评估】": "",
            "TP": total_tp,
            "FP": total_fp,
            "FP(near_ref)": total_fp_near_ref,
            "FP(far)": total_fp_far,
            "FN": total_fn,
            "Precision": f"{precision:.3f}",
            "Recall": f"{recall:.3f}",
            "F1": f"{f1:.3f}",
            "负例文件数": negative_files,
            "负例出现FP的文件数": negative_fp_files,
            "负例FP片段总数": negative_fp_segments,
        }
    )

    logger.note(dict_to_lines(info_dict, key_prefix="* "))

    return results


class TemplateMatcherArgParser:
    """模板匹配器命令行参数解析器。"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GTAV 音频模板匹配器")
        self._add_arguments()

    def _add_arguments(self):
        """添加命令行参数。"""
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="测试模式：使用缓存的测试音频文件测试模板匹配效果",
        )
        self.parser.add_argument(
            "-t",
            "--templates-dir",
            type=str,
            default=None,
            help=f"模板目录（默认: {TEMPLATES_DIR}）",
        )
        self.parser.add_argument(
            "-f",
            "--template-file",
            type=str,
            default=None,
            help="单个模板文件路径",
        )
        self.parser.add_argument(
            "-T",
            "--threshold",
            type=float,
            default=MATCH_THRESHOLD,
            help=f"匹配阈值（0-1，默认: {MATCH_THRESHOLD}）",
        )
        self.parser.add_argument(
            "-c",
            "--cooldown-ms",
            type=float,
            default=COOLDOWN_MS,
            help=f"冷却时间（毫秒，默认: {COOLDOWN_MS}）",
        )
        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=None,
            help="运行时长（秒），不指定则持续运行",
        )
        self.parser.add_argument(
            "-n",
            "--device-name",
            type=str,
            default=AUDIO_DEVICE_NAME,
            help=f"音频设备名称（默认: {AUDIO_DEVICE_NAME}）",
        )
        self.parser.add_argument(
            "-r",
            "--sample-rate",
            type=int,
            default=SAMPLE_RATE,
            help=f"采样率（默认: {SAMPLE_RATE}）",
        )
        self.parser.add_argument(
            "-w",
            "--window-ms",
            type=int,
            default=WINDOW_MS,
            help=f"窗口时长（毫秒，默认: {WINDOW_MS}）",
        )
        self.parser.add_argument(
            "-i",
            "--interval-ms",
            type=float,
            default=100,
            help="匹配间隔（毫秒，默认: 100）",
        )
        self.parser.add_argument(
            "-s",
            "--sounds-dir",
            type=str,
            default=None,
            help=f"测试音频目录（仅用于 --test 模式，默认: {TEST_SOUNDS_DIR}）",
        )
        self.parser.add_argument(
            "-p",
            "--prefix",
            type=str,
            default=TEMPLATE_PREFIX,
            help=f"模板文件前缀过滤（默认: {TEMPLATE_PREFIX}）",
        )

        # 特征权重参数
        self.parser.add_argument(
            "--weight-time",
            type=float,
            default=0.25,
            help="时域相关性权重（默认: 0.25）",
        )
        self.parser.add_argument(
            "--weight-mel",
            type=float,
            default=0.35,
            help="梅尔频谱权重（默认: 0.35）",
        )
        self.parser.add_argument(
            "--weight-env",
            type=float,
            default=0.25,
            help="能量包络权重（默认: 0.25）",
        )
        self.parser.add_argument(
            "--weight-zcr",
            type=float,
            default=0.05,
            help="零交叉率权重（默认: 0.05）",
        )
        self.parser.add_argument(
            "--weight-centroid",
            type=float,
            default=0.10,
            help="频谱质心权重（默认: 0.10）",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数。"""
        return self.parser.parse_args()


def main():
    """命令行入口。"""
    args = TemplateMatcherArgParser().parse()

    # 测试模式
    if args.test:
        templates_dir = Path(args.templates_dir) if args.templates_dir else None
        sounds_dir = Path(args.sounds_dir) if args.sounds_dir else None
        test_templates(
            templates_dir=templates_dir,
            sounds_dir=sounds_dir,
            threshold=args.threshold,
            prefix=args.prefix,
        )
        return

    # 创建录制器
    recorder = SoundRecorder(
        device_name=args.device_name,
        sample_rate=args.sample_rate,
        window_ms=args.window_ms,
    )

    # 创建匹配器（使用较低采样率以提高匹配速度，并设置特征权重）
    matcher = TemplateMatcher(
        threshold=args.threshold,
        target_sample_rate=REALTIME_SAMPLE_RATE,
        cooldown_ms=args.cooldown_ms,
        weight_time_corr=args.weight_time,
        weight_mel_spec=args.weight_mel,
        weight_energy_env=args.weight_env,
        weight_zcr=args.weight_zcr,
        weight_spectral_centroid=args.weight_centroid,
    )

    # 创建实时匹配器
    realtime_matcher = RealtimeMatcher(
        recorder=recorder,
        matcher=matcher,
        match_interval_ms=args.interval_ms,
    )

    # 加载模板
    if args.template_file:
        loader = TemplateLoader()
        template = loader.load_file(Path(args.template_file))
        if template:
            realtime_matcher.add_template(template)
    elif args.templates_dir:
        realtime_matcher.load_templates(Path(args.templates_dir))
    else:
        # 尝试从默认目录加载
        if TEMPLATES_DIR.exists():
            realtime_matcher.load_templates(TEMPLATES_DIR)
        else:
            logger.warn(f"默认模板目录不存在: {TEMPLATES_DIR}")
            logger.note("请使用 -t 指定模板目录或 -f 指定模板文件")
            return

    if realtime_matcher.matcher.template_count == 0:
        logger.warn("没有加载任何模板")
        return

    # 运行
    realtime_matcher.run(duration=args.duration)


if __name__ == "__main__":
    main()

    # Case: 测试
    # python -m gtaz.audios.signals --test

    # Case: 持续匹配
    # python -m gtaz.audios.signals

    # Case: 调整阈值和冷却时间
    # python -m gtaz.audios.signals -T 0.7 -c 3000

    # Case: 定时运行 60 秒
    # python -m gtaz.audios.signals -d 60
