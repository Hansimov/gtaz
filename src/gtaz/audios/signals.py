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


logger = TCLogger(name="TemplateMatcher", use_prefix=True, use_prefix_ms=True)


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
MATCH_THRESHOLD = 0.6
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
            logger.okay(f"已加载模板: {template}")
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
    """

    def __init__(self, target_sample_rate: int = REALTIME_SAMPLE_RATE):
        """
        初始化特征提取器。

        :param target_sample_rate: 目标采样率
        """
        self.target_sample_rate = target_sample_rate

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

    def extract(
        self,
        data: np.ndarray,
        sample_rate: int,
        compute_centered: bool = False,
    ) -> AudioFeatures:
        """
        提取音频特征。

        :param data: 原始音频数据
        :param sample_rate: 原始采样率
        :param compute_centered: 是否计算零均值化数据

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

        return AudioFeatures(
            data=processed_data,
            sample_rate=self.target_sample_rate,
            energy=energy,
            mean=data_mean,
            centered_data=centered_data,
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
        # 计算匹配位置的时间（基于原始音频采样率）
        if self.original_sample_rate > 0 and self.sample_rate > 0:
            # 将 position 从 target_sample_rate 转换回 original_sample_rate
            original_position = (
                self.position * self.original_sample_rate / self.sample_rate
            )

            # 计算原始音频中的时间
            start_seconds = original_position / self.original_sample_rate
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
    音频模板匹配器（FFT 加速版）。

    使用 FFT 加速的归一化互相关算法进行模板匹配，支持实时音频流。
    主要优化：
    - 预计算模板的 FFT 和能量
    - 使用 FFT 进行快速互相关
    - 降采样以减少计算量
    - 批量处理多模板匹配
    """

    def __init__(
        self,
        templates: list[AudioTemplate] = None,
        threshold: float = MATCH_THRESHOLD,
        target_sample_rate: int = REALTIME_SAMPLE_RATE,
        cooldown_ms: float = COOLDOWN_MS,
        features_extractor: FeaturesExtractor = None,
    ):
        """
        初始化模板匹配器。

        :param templates: 音频模板列表
        :param threshold: 匹配阈值（0-1）
        :param target_sample_rate: 目标采样率（用于重采样，较低采样率可提高速度）
        :param cooldown_ms: 检测冷却时间（毫秒）
        :param features_extractor: 特征提取器，默认自动创建
        """
        self.threshold = threshold
        self.target_sample_rate = target_sample_rate
        self.cooldown_ms = cooldown_ms

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
            f"已添加模板: {template.name} ({len(features.data)} samples @ {features.sample_rate}Hz)"
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
        min_energy_ratio = 0.1  # 提高到 10%
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

    def match_single(
        self,
        window_data: np.ndarray,
        template_name: str,
        sample_rate: int = None,
        check_cooldown: bool = True,
        preprocessed_window: np.ndarray = None,
    ) -> MatchResult:
        """
        对单个模板进行匹配。

        :param window_data: 窗口音频数据
        :param template_name: 模板名称
        :param sample_rate: 窗口采样率，默认使用 target_sample_rate
        :param check_cooldown: 是否检查冷却时间
        :param preprocessed_window: 已预处理的窗口数据（可选，用于批量匹配时避免重复预处理）

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
                window_data, sample_rate, compute_centered=True
            )

        # 计算相关性（使用 FFT 加速）
        score, position = self._compute_correlation_fft(
            window_features, template_features
        )

        # 判断是否匹配
        matched = score >= self.threshold

        # 更新冷却时间
        if matched:
            self._last_match_times[template_name] = current_time

        # 计算置信度（将分数映射到 0-1 范围，以阈值为中点）
        if score >= self.threshold:
            confidence = 0.5 + 0.5 * (score - self.threshold) / (
                1 - self.threshold + 1e-10
            )
        else:
            confidence = 0.5 * score / (self.threshold + 1e-10)

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
            score=score,
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
    ) -> list[MatchResult]:
        """
        对所有模板进行匹配（优化版：只预处理一次窗口数据）。

        :param window_data: 窗口音频数据
        :param sample_rate: 窗口采样率
        :param check_cooldown: 是否检查冷却时间

        :return: 所有模板的匹配结果列表
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        # 预处理窗口数据（只做一次）
        preprocessed_window = self.features_extractor.extract(
            window_data, sample_rate, compute_centered=True
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
            f"cooldown_ms={self.cooldown_ms})"
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
    total_match_time = 0.0

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

        # 执行匹配（禁用冷却）并计时
        start_time = time.perf_counter()
        match_results = matcher.match_all(
            data,
            sample_rate=sample_rate,
            check_cooldown=False,
        )
        match_time = (time.perf_counter() - start_time) * 1000
        total_match_time += match_time

        # 将匹配位置补偿回原始音频位置（加上跳过的 5 秒）
        for result in match_results:
            # position 是基于 target_sample_rate 的，需要加上对应的偏移
            skip_samples_resampled = int(skip_seconds * result.sample_rate)
            result.position += skip_samples_resampled

        # 记录结果
        rel_path = test_file.relative_to(sounds_dir)
        results[str(rel_path)] = match_results

        # 按分数排序
        sorted_results = sorted(match_results, key=lambda r: r.score, reverse=True)
        best = sorted_results[0] if sorted_results else None

        # 保存结果到同名 JSON 文件（添加 _match 后缀）
        json_path = test_file.with_name(test_file.stem + "_match.json")
        json_data = {
            "file": test_file.name,
            "match_time_ms": round(match_time, 2),
            "best_match": best.to_dict() if best else None,
            "all_matches": [r.to_dict() for r in sorted_results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 输出结果
        logger.file(f"\n测试文件: {rel_path}")

        # 输出匹配结果（低分用 warn，高分用 okay）
        LOW_SCORE_THRESHOLD = 0.75
        for result in sorted_results:
            result_dict = result.to_dict()
            time_info = f"[{result_dict['start_time']} ~ {result_dict['end_time']}] ({result_dict['duration']}s)"

            if result.score < LOW_SCORE_THRESHOLD:
                # 低分结果：整行用 warn
                logger.warn(
                    f"  ⚠ {result.template_name}: {result.score:.3f} {time_info}"
                )
            elif result.matched:
                # 高分匹配：使用颜色样式
                template_str = logstr.file(result.template_name)
                score_str = logstr.okay(f"{result.score:.3f}")
                time_str = logstr.mesg(time_info)
                logger.okay(f"  ✓ {template_str}: {score_str} {time_str}")
            else:
                logger.mesg(
                    f"  · {result.template_name}: {result.score:.3f} {time_info}"
                )

        # 输出最佳匹配和 JSON 保存信息
        if best and best.matched:
            best_dict = best.to_dict()
            best_template = logstr.file(best.template_name)
            best_score = logstr.okay(f"{best.score:.3f}")
            best_time = logstr.mesg(
                f"[{best_dict['start_time']}s ~ {best_dict['end_time']}s]"
            )
            logger.okay(f"  → 最佳匹配: {best_template} ({best_score}) {best_time}")
        logger.file(f"  → 结果已保存: {json_path.name}")

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

    # 创建匹配器（使用较低采样率以提高匹配速度）
    matcher = TemplateMatcher(
        threshold=args.threshold,
        target_sample_rate=REALTIME_SAMPLE_RATE,
        cooldown_ms=args.cooldown_ms,
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
