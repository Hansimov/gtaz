"""实时音频模板匹配检测模块

结合 sounds.py 的实时音频采集和 signals_v3.py 的模板匹配算法，
实现实时监控音频流并检测匹配事件。
"""

import argparse
import pickle
import numpy as np
import time

from pathlib import Path
from typing import Optional, Callable, Any
from tclogger import TCLogger, logstr

from .sounds import (
    SoundRecorder,
    AUDIO_DEVICE_NAME,
    SAMPLE_RATE,
    CHANNELS,
    WINDOW_MS,
    VOLUME_CHARS,
    VOLUME_BITS,
    VOLUME_GAIN,
    SAMPLES_PER_GROUP,
    VOLUME_SAMPLE_INTERVAL_MS,
)
from .signals_v3 import (
    TemplateLoader,
    AudioDataSamplesUnifier,
    FeatureExtractor,
    FeatureMatcher,
    TemplateAligner,
    TemplateFuser,
    Feature,
    MatchResult,
    MatchCandidate,
    TEMPLATES_DIR,
    TEMPLATE_REGEX,
    UNIFIED_SAMPLE_RATE,
    MATCH_WINDOW_MS,
    MATCH_STEP_MS,
    MATCH_GATE,
    CANDIDATE_MIN_OFFSET_MS,
)

logger = TCLogger(
    name="AudioDetector",
    use_prefix=True,
    use_prefix_ms=True,
    use_file=True,
    file_mode="w",
)

# 当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 模板特征缓存文件
TEMPLATE_FEATURE_CACHE = MODULE_DIR / "template_feature.pkl"

# 检测间隔（毫秒）- 每隔多久执行一次匹配
DETECT_INTERVAL_MS = 200

# 绝对音量阈值（音量百分比）- 平均音量低于此值时不进行匹配
# 这是日志中显示的音量数字，基于最近采样的 RMS 值
VOLUME_THRESHOLD_PERCENT = 25


class TemplateFeatureManager:
    """模板特征管理器

    负责加载、融合和缓存模板特征。
    首次运行时计算并保存，后续运行直接加载缓存。
    """

    def __init__(
        self,
        templates_dir: Path = TEMPLATES_DIR,
        template_regex: str = TEMPLATE_REGEX,
        cache_path: Path = TEMPLATE_FEATURE_CACHE,
    ):
        self.templates_dir = templates_dir
        self.template_regex = template_regex
        self.cache_path = cache_path

        # 组件
        self.template_loader = TemplateLoader(templates_dir, template_regex)
        self.unifier = AudioDataSamplesUnifier()
        self.feature_extractor = FeatureExtractor()
        self.aligner = TemplateAligner()
        self.fuser = TemplateFuser(self.feature_extractor)

        # 融合后的模板特征
        self.fused_feature: Optional[Feature] = None

    def _compute_templates_hash(self) -> str:
        """计算模板文件的哈希值（用于检测模板是否变化）"""
        import hashlib

        self.template_loader.scan_templates()
        hash_content = ""
        for file_path in sorted(self.template_loader.template_files):
            stat = file_path.stat()
            hash_content += f"{file_path.name}:{stat.st_size}:{stat.st_mtime};"
        return hashlib.md5(hash_content.encode()).hexdigest()

    def _load_cache(self) -> Optional[Feature]:
        """从缓存加载模板特征"""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # # 检查模板是否变化
            # cached_hash = cache_data.get("hash", "")
            # current_hash = self._compute_templates_hash()

            # if cached_hash != current_hash:
            #     logger.note("模板文件已变化，需要重新计算特征")
            #     return None

            logger.okay(f"从缓存加载模板特征: {self.cache_path}")
            return cache_data["feature"]

        except Exception as e:
            logger.warn(f"加载缓存失败: {e}")
            return None

    def _save_cache(self, feature: Feature):
        """保存模板特征到缓存"""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "hash": self._compute_templates_hash(),
                "feature": feature,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.okay(f"模板特征已缓存: {self.cache_path}")
        except Exception as e:
            logger.warn(f"保存缓存失败: {e}")

    def _compute_feature(self) -> Feature:
        """计算模板特征"""
        logger.note("计算模板特征...")

        # 扫描并加载模板
        self.template_loader.scan_templates()
        template_data = self.template_loader.load_templates()

        if len(template_data) == 0:
            raise ValueError("没有找到模板文件")

        # 统一采样率
        unified_templates = self.unifier.unify(template_data)

        # 对齐模板
        aligned_templates, offsets = self.aligner.align_templates(unified_templates)

        # 找到中心融合区域
        region_start, region_end = self.aligner.find_center_region(
            unified_templates, offsets
        )

        # 提取对齐后的区域
        extracted_templates, weights = self.aligner.extract_aligned_region(
            unified_templates, offsets, region_start, region_end
        )

        # 融合模板
        fused_feature = self.fuser.fuse_templates(extracted_templates, weights)

        logger.okay(
            f"模板特征计算完成, 维度: {fused_feature.data.shape}, "
            f"时长: {fused_feature.duration_ms:.1f}ms"
        )

        return fused_feature

    def get_feature(self, force_recompute: bool = False) -> Feature:
        """获取模板特征（优先从缓存加载）"""
        if not force_recompute:
            cached = self._load_cache()
            if cached is not None:
                self.fused_feature = cached
                return cached

        # 计算并缓存
        self.fused_feature = self._compute_feature()
        self._save_cache(self.fused_feature)
        return self.fused_feature


class AudioDetector:
    """实时音频检测器

    使用 SoundRecorder 采集音频，使用 FeatureMatcher 进行模板匹配。
    """

    def __init__(
        self,
        device_name: str = AUDIO_DEVICE_NAME,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        window_ms: int = WINDOW_MS,
        gate: float = MATCH_GATE,
        detect_interval_ms: int = DETECT_INTERVAL_MS,
    ):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_ms = window_ms
        self.gate = gate
        self.detect_interval_ms = detect_interval_ms

        # 模板特征管理器
        self.template_manager = TemplateFeatureManager()
        self.template_feature: Optional[Feature] = None

        # 音频录制器
        self.recorder = SoundRecorder(
            device_name=device_name,
            sample_rate=sample_rate,
            channels=channels,
            window_ms=window_ms,
        )

        # 特征匹配器（延迟初始化）
        self.matcher: Optional[FeatureMatcher] = None

        # 音频统一化器
        self.unifier = AudioDataSamplesUnifier()

        # 检测回调
        self._on_match_callback: Optional[Callable[[float, MatchResult], Any]] = None

        # 最后检测时间
        self._last_detect_time: float = 0

        # 音量显示相关
        self._sample_count = 0
        self._group_volumes: list[int] = []
        self._line_buffer: list[str] = []
        self._current_tick_detected = False  # 当前 tick 是否检测到信号

        # 组内最高分数追踪
        self._group_max_score = 0.0
        self._group_max_score_index = -1

    def _is_first_in_group(self) -> bool:
        """判断是否是组内第一个。"""
        return self._sample_count % SAMPLES_PER_GROUP == 0

    def _is_last_in_group(self) -> bool:
        """判断是否是组内最后一个。"""
        return (self._sample_count + 1) % SAMPLES_PER_GROUP == 0

    def _log_volume_char(self, volume_char: str, highlight: bool = False):
        """输出音量字符。"""
        # 只使用传入的高亮状态（当前 tick 检测到才高亮）
        display_char = logstr.okay(volume_char) if highlight else volume_char

        if self._is_first_in_group():
            self._line_buffer = []
            logger.mesg(display_char, end="")
        else:
            logger.mesg(display_char, use_prefix=False, end="")
        self._line_buffer.append(volume_char)

    @staticmethod
    def _calculate_stats(volumes: list[int]) -> tuple[int, float, int]:
        """计算音量统计信息。"""
        if not volumes:
            return 0, 0.0, 0
        return min(volumes), sum(volumes) / len(volumes), max(volumes)

    def _log_group_stats(self, score: float = None):
        """输出一组音量的统计信息（显示组内最高分数）。"""
        if self._group_volumes:
            min_vol, avg_vol, max_vol = self._calculate_stats(self._group_volumes)
            vol_strs = [
                logstr.mesg(f"{round(v):2d}") for v in [min_vol, avg_vol, max_vol]
            ]
            vol_line = "/".join(vol_strs)
            # 使用组内最高分数而非最后一次分数
            display_score = self._group_max_score
            display_index = self._group_max_score_index
            if display_score >= self.gate:
                score_str = logstr.okay(f"{display_score:.3f}✓")
                if display_index >= 0:
                    score_str += logstr.hint(f" @{display_index}")
            else:
                score_str = logstr.file(f"{display_score:.3f}")
            logger.note(f" [{vol_line}] 分数={score_str}", use_prefix=False)
        else:
            logger.note("", use_prefix=False)

    def set_on_match_callback(self, callback: Callable[[float, MatchResult], Any]):
        """设置匹配回调函数

        :param callback: 回调函数，接收 (score, match_result) 参数
        """
        self._on_match_callback = callback

    def initialize(self, force_recompute: bool = False) -> bool:
        """初始化检测器（加载模板特征）

        :param force_recompute: 是否强制重新计算模板特征
        :return: 是否成功初始化
        """
        try:
            self.template_feature = self.template_manager.get_feature(force_recompute)

            # 更新窗口大小为模板时长
            match_window_ms = self.template_feature.duration_ms

            # 输出模板特征信息
            logger.note(
                f"模板特征: duration={match_window_ms:.1f}ms, "
                f"shape={self.template_feature.data.shape}, "
                f"volume={self.template_feature.volume:.6f}"
            )

            # 创建匹配器
            self.matcher = FeatureMatcher(
                template_feature=self.template_feature,
                window_ms=match_window_ms,
                step_ms=MATCH_STEP_MS,
                gate=self.gate,
                min_offset_ms=CANDIDATE_MIN_OFFSET_MS,
                sample_rate=UNIFIED_SAMPLE_RATE,
            )

            # 计算匹配窗口采样点数
            window_samples = int(match_window_ms * UNIFIED_SAMPLE_RATE / 1000)
            logger.note(
                f"匹配参数: window_samples={window_samples}, "
                f"step_ms={MATCH_STEP_MS}, gate={self.gate}"
            )

            logger.okay("检测器初始化完成")
            return True

        except Exception as e:
            logger.warn(f"初始化失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _preprocess_audio(self, data: np.ndarray) -> np.ndarray:
        """预处理音频数据（转单声道、统一采样率）"""
        # 转单声道
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        elif len(data.shape) > 1:
            data = data.flatten()

        # 统一采样率
        data = self.unifier.unify_single(self.sample_rate, data)
        return data

    def detect(self, debug: bool = False) -> tuple[float, Optional[MatchResult]]:
        """执行一次检测（快速模式：只对最近的窗口进行单次匹配）

        :param debug: 是否输出调试信息
        :return: (匹配分数, 匹配结果)，无匹配时分数为 0
        """
        if self.matcher is None:
            logger.warn("检测器未初始化")
            return 0.0, None

        # 获取窗口数据
        window_data = self.recorder.buffer.get_window_data()
        if window_data is None or len(window_data) == 0:
            return 0.0, None

        # 预处理（转单声道、统一采样率）
        processed_data = self._preprocess_audio(window_data)

        # 计算需要的窗口采样点数
        window_samples = int(self.matcher.window_ms * self.matcher.sample_rate / 1000)

        # 检查数据是否足够
        if len(processed_data) < window_samples:
            if debug:
                logger.warn(f"数据长度不足: {len(processed_data)} < {window_samples}")
            return 0.0, None

        # 只取最后一个窗口的数据进行检测（快速模式）
        last_window_data = processed_data[-window_samples:]

        # 提取特征
        window_feature = self.matcher.feature_extractor.extract(last_window_data)

        # 调试：输出音量对比
        if debug:
            template_vol = self.matcher.template_feature.volume
            test_vol = window_feature.volume
            vol_ratio = test_vol / template_vol if template_vol > 1e-10 else 0
            logger.mesg(
                f"  [DEBUG] 音量对比: template={template_vol:.6f}, "
                f"test={test_vol:.6f}, ratio={vol_ratio:.4f}"
            )

        # 计算相似度
        score = self.matcher.compute_similarity(
            self.matcher.template_feature, window_feature
        )

        # 创建匹配结果
        total_duration_ms = len(processed_data) / self.matcher.sample_rate * 1000
        candidates = []
        if score >= self.matcher.gate:
            start_ms = total_duration_ms - self.matcher.window_ms
            candidates.append(
                MatchCandidate(
                    start_ms=start_ms,
                    end_ms=total_duration_ms,
                    score=score,
                    start_sample=len(processed_data) - window_samples,
                    end_sample=len(processed_data),
                )
            )

        result = MatchResult(
            test_file="",
            total_duration_ms=total_duration_ms,
            sample_rate=self.matcher.sample_rate,
            candidates=candidates,
            match_window_ms=self.matcher.window_ms,
            match_step_ms=self.matcher.step_ms,
            match_gate=self.matcher.gate,
            candidate_min_offset_ms=self.matcher.min_offset_ms,
        )

        return score, result

    def detect_sliding(
        self, return_max_raw_score: bool = True, debug: bool = False
    ) -> tuple[float, Optional[MatchResult]]:
        """执行一次检测（滑动窗口模式：扫描整个缓冲区，较慢）

        :param return_max_raw_score: 是否返回原始最高分数（即使未超过阈值）
        :param debug: 是否输出调试信息
        :return: (最高匹配分数, 匹配结果)，无匹配时分数为 0
        """
        if self.matcher is None:
            logger.warn("检测器未初始化")
            return 0.0, None

        # 获取窗口数据
        window_data = self.recorder.buffer.get_window_data()
        if window_data is None or len(window_data) == 0:
            return 0.0, None

        # 预处理
        processed_data = self._preprocess_audio(window_data)

        # 调试：输出数据信息
        if debug:
            logger.mesg(
                f"  [DEBUG] processed_data: len={len(processed_data)}, "
                f"max={np.max(np.abs(processed_data)):.6f}, "
                f"rms={np.sqrt(np.mean(processed_data**2)):.6f}"
            )

        # 执行匹配（获取原始最高分数）
        if return_max_raw_score:
            max_score, result = self._match_with_raw_score(processed_data, debug=debug)
        else:
            result = self.matcher.match(processed_data)
            # 计算最高分数
            if result.candidates:
                max_score = max(c.score for c in result.candidates)
            else:
                max_score = 0.0

        return max_score, result

    def _match_with_raw_score(
        self, test_data: np.ndarray, debug: bool = False
    ) -> tuple[float, MatchResult]:
        """匹配并返回原始最高分数（即使未超过阈值）"""
        total_duration_ms = len(test_data) / self.matcher.sample_rate * 1000

        # 计算滑动窗口参数
        window_samples = int(self.matcher.window_ms * self.matcher.sample_rate / 1000)
        step_samples = int(self.matcher.step_ms * self.matcher.sample_rate / 1000)

        # 检查数据长度是否足够
        if len(test_data) < window_samples:
            if debug:
                logger.warn(
                    f"数据长度不足: {len(test_data)} < {window_samples} (需要 {self.matcher.window_ms}ms)"
                )
            # 数据不足，返回空结果
            return 0.0, MatchResult(
                test_file="",
                total_duration_ms=total_duration_ms,
                sample_rate=self.matcher.sample_rate,
                candidates=[],
            )

        # 存储所有候选和所有分数
        all_candidates = []
        all_scores = []

        # 调试：记录第一个窗口的音量信息
        first_window_logged = False

        # 滑动窗口匹配
        start_sample = 0
        while start_sample + window_samples <= len(test_data):
            # 提取窗口特征
            window_feature = self.matcher.feature_extractor.extract_window(
                test_data, start_sample, window_samples
            )

            # 调试：输出第一个窗口的音量对比
            if debug and not first_window_logged:
                template_vol = self.matcher.template_feature.volume
                test_vol = window_feature.volume
                vol_ratio = test_vol / template_vol if template_vol > 1e-10 else 0
                logger.mesg(
                    f"  [DEBUG] 音量对比: template={template_vol:.6f}, "
                    f"test={test_vol:.6f}, ratio={vol_ratio:.4f}"
                )
                first_window_logged = True

            # 计算相似度
            score = self.matcher.compute_similarity(
                self.matcher.template_feature, window_feature
            )
            all_scores.append(score)

            # 如果超过阈值，添加候选
            if score >= self.matcher.gate:
                start_ms = start_sample / self.matcher.sample_rate * 1000
                end_ms = (
                    (start_sample + window_samples) / self.matcher.sample_rate * 1000
                )
                candidate = MatchCandidate(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=score,
                    start_sample=start_sample,
                    end_sample=start_sample + window_samples,
                )
                all_candidates.append(candidate)

            start_sample += step_samples

        # 计算最高原始分数
        max_raw_score = max(all_scores) if all_scores else 0.0

        # 非极大值抑制
        filtered_candidates = self.matcher._non_max_suppression(all_candidates)

        result = MatchResult(
            test_file="",
            total_duration_ms=total_duration_ms,
            sample_rate=self.matcher.sample_rate,
            candidates=filtered_candidates,
            match_window_ms=self.matcher.window_ms,
            match_step_ms=self.matcher.step_ms,
            match_gate=self.matcher.gate,
            candidate_min_offset_ms=self.matcher.min_offset_ms,
        )

        return max_raw_score, result

    def detect_loop(self, duration: float = None, debug: bool = False):
        """持续检测循环

        :param duration: 检测时长（秒），None 表示持续检测
        :param debug: 是否输出调试信息（首次检测时输出）
        """
        if self.matcher is None:
            logger.warn("检测器未初始化")
            return

        # 启动音频流
        if not self.recorder.start_stream():
            logger.warn("无法启动音频流")
            return

        logger.note(
            f"开始实时检测 (阈值: {self.gate}, 间隔: {self.detect_interval_ms}ms, 音量阈值: {VOLUME_THRESHOLD_PERCENT})"
        )
        if duration:
            logger.note(f"检测时长: {duration}秒")

        start_time = time.time()
        self._sample_count = 0
        self._group_volumes = []
        self._current_tick_detected = False
        self._group_max_score = 0.0
        self._group_max_score_index = -1
        detect_count = 0
        first_detect = True

        try:
            while True:
                # 检查时长
                if duration and (time.time() - start_time) >= duration:
                    # 输出最后一组统计
                    if self._sample_count % SAMPLES_PER_GROUP != 0:
                        self._log_group_stats()
                    logger.note("检测时长已到")
                    break

                # 组开始时重置组内最高分数
                if self._is_first_in_group():
                    self._group_max_score = 0.0
                    self._group_max_score_index = -1

                # 重置当前 tick 的检测状态
                self._current_tick_detected = False

                # 检测间隔到达时执行检测（在输出音量字符之前）
                current_time = time.time()
                if (
                    current_time - self._last_detect_time
                ) * 1000 >= self.detect_interval_ms:
                    self._last_detect_time = current_time

                    # 计算最近采样的平均音量
                    recent_avg_volume = (
                        sum(self._group_volumes) / len(self._group_volumes)
                        if self._group_volumes
                        else 0
                    )

                    # 只有音量足够时才进行检测
                    if recent_avg_volume >= VOLUME_THRESHOLD_PERCENT:
                        # 首次检测时输出调试信息
                        score, result = self.detect(debug=(debug and first_detect))
                        first_detect = False
                        detect_count += 1

                        # 更新组内最高分数
                        tick_in_group = self._sample_count % SAMPLES_PER_GROUP
                        if score > self._group_max_score:
                            self._group_max_score = score
                            self._group_max_score_index = tick_in_group

                        # 如果匹配成功，设置当前 tick 检测状态并调用回调
                        if score >= self.gate:
                            self._current_tick_detected = True
                            if self._on_match_callback and result:
                                self._on_match_callback(score, result)
                    else:
                        # 音量不足，跳过检测
                        first_detect = False

                # 获取当前音量并显示（检测状态决定是否高亮）
                volume_percent = self.recorder.get_volume_percent()
                self._group_volumes.append(volume_percent)
                volume_char = self.recorder.get_volume_char(volume_percent)
                self._log_volume_char(
                    volume_char, highlight=self._current_tick_detected
                )

                # 组结束时输出统计和分数
                if self._is_last_in_group():
                    self._log_group_stats()
                    self._group_volumes = []

                self._sample_count += 1
                time.sleep(VOLUME_SAMPLE_INTERVAL_MS / 1000)

        except KeyboardInterrupt:
            # 输出最后一组统计
            if self._sample_count % SAMPLES_PER_GROUP != 0:
                self._log_group_stats()
            logger.note("\n检测到 Ctrl+C，正在退出...")

        finally:
            self.recorder.stop_stream()
            logger.okay(f"检测结束，共执行 {detect_count} 次检测")

    def detect_then_stop(
        self,
        timeout: float = None,
        on_match: Callable[[float, MatchResult], Any] = None,
    ) -> tuple[bool, float, Optional[MatchResult]]:
        """检测直到匹配成功或超时

        :param timeout: 超时时间（秒），None 表示无超时
        :param on_match: 匹配成功时的回调函数
        :return: (是否匹配成功, 最高分数, 匹配结果)
        """
        if self.matcher is None:
            logger.warn("检测器未初始化")
            return False, 0.0, None

        # 启动音频流
        if not self.recorder.start_stream():
            logger.warn("无法启动音频流")
            return False, 0.0, None

        logger.note(
            f"开始检测 (阈值: {self.gate}, 音量阈值: {VOLUME_THRESHOLD_PERCENT})，匹配到即停止"
        )
        if timeout:
            logger.note(f"超时时间: {timeout}秒")

        start_time = time.time()
        self._sample_count = 0
        self._group_volumes = []
        self._current_tick_detected = False
        self._group_max_score = 0.0
        self._group_max_score_index = -1
        matched = False
        final_score = 0.0
        final_result = None

        try:
            while True:
                # 检查超时
                if timeout and (time.time() - start_time) >= timeout:
                    # 输出最后一组统计
                    if self._sample_count % SAMPLES_PER_GROUP != 0:
                        self._log_group_stats()
                    logger.note("检测超时")
                    break

                # 组开始时重置组内最高分数
                if self._is_first_in_group():
                    self._group_max_score = 0.0
                    self._group_max_score_index = -1

                # 重置当前 tick 的检测状态
                self._current_tick_detected = False

                # 检测间隔到达时执行检测（在输出音量字符之前）
                current_time = time.time()
                if (
                    current_time - self._last_detect_time
                ) * 1000 >= self.detect_interval_ms:
                    self._last_detect_time = current_time

                    # 计算最近采样的平均音量
                    recent_avg_volume = (
                        sum(self._group_volumes) / len(self._group_volumes)
                        if self._group_volumes
                        else 0
                    )

                    # 只有音量足够时才进行检测
                    if recent_avg_volume >= VOLUME_THRESHOLD_PERCENT:
                        score, result = self.detect()

                        # 更新组内最高分数
                        tick_in_group = self._sample_count % SAMPLES_PER_GROUP
                        if score > self._group_max_score:
                            self._group_max_score = score
                            self._group_max_score_index = tick_in_group

                        # 检查是否匹配
                        if score >= self.gate:
                            self._current_tick_detected = True
                            # 先输出当前高亮的音量字符
                            volume_percent = self.recorder.get_volume_percent()
                            self._group_volumes.append(volume_percent)
                            volume_char = self.recorder.get_volume_char(volume_percent)
                            self._log_volume_char(volume_char, highlight=True)
                            # 输出当前组统计
                            if self._sample_count % SAMPLES_PER_GROUP != 0:
                                self._log_group_stats()
                            elapsed = current_time - start_time
                            logger.okay(
                                f"[{elapsed:.1f}s] 匹配成功! 分数={logstr.okay(f'{score:.4f}')}"
                            )
                            matched = True
                            final_score = score
                            final_result = result

                            # 调用回调
                            if on_match and result:
                                on_match(score, result)

                            break

                # 获取当前音量并显示
                volume_percent = self.recorder.get_volume_percent()
                self._group_volumes.append(volume_percent)
                volume_char = self.recorder.get_volume_char(volume_percent)
                self._log_volume_char(
                    volume_char, highlight=self._current_tick_detected
                )

                # 组结束时输出统计和分数
                if self._is_last_in_group():
                    self._log_group_stats()
                    self._group_volumes = []

                self._sample_count += 1
                time.sleep(VOLUME_SAMPLE_INTERVAL_MS / 1000)

        except KeyboardInterrupt:
            # 输出最后一组统计
            if self._sample_count % SAMPLES_PER_GROUP != 0:
                self._log_group_stats()
            logger.note("\n检测到 Ctrl+C，正在退出...")

        finally:
            self.recorder.stop_stream()

        return matched, final_score, final_result


class AudioDetectorArgParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="实时音频模板匹配检测")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=None,
            help="检测时长（秒），不指定则持续检测",
        )
        self.parser.add_argument(
            "-g",
            "--gate",
            type=float,
            default=MATCH_GATE,
            help=f"匹配阈值（默认: {MATCH_GATE}）",
        )
        self.parser.add_argument(
            "-i",
            "--interval",
            type=int,
            default=DETECT_INTERVAL_MS,
            help=f"检测间隔（毫秒，默认: {DETECT_INTERVAL_MS}）",
        )
        self.parser.add_argument(
            "-n",
            "--device-name",
            type=str,
            default=AUDIO_DEVICE_NAME,
            help=f"音频设备名称（默认: {AUDIO_DEVICE_NAME}）",
        )
        self.parser.add_argument(
            "-w",
            "--window-ms",
            type=int,
            default=WINDOW_MS,
            help=f"缓冲区窗口时长（毫秒，默认: {WINDOW_MS}）",
        )
        self.parser.add_argument(
            "-s",
            "--stop-on-match",
            action="store_true",
            help="匹配到即停止",
        )
        self.parser.add_argument(
            "-r",
            "--recompute",
            action="store_true",
            help="强制重新计算模板特征",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="输出调试信息（首次检测时显示音量对比）",
        )

    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()


def main():
    """命令行入口"""
    args = AudioDetectorArgParser().parse()

    # 创建检测器
    detector = AudioDetector(
        device_name=args.device_name,
        window_ms=args.window_ms,
        gate=args.gate,
        detect_interval_ms=args.interval,
    )

    # 初始化
    if not detector.initialize(force_recompute=args.recompute):
        logger.warn("检测器初始化失败")
        return

    # 运行检测
    if args.stop_on_match:
        matched, score, result = detector.detect_then_stop(timeout=args.duration)
        if matched:
            logger.okay(f"检测完成，匹配分数: {score:.4f}")
        else:
            logger.note("未匹配到目标音频")
    else:
        detector.detect_loop(duration=args.duration, debug=args.debug)


if __name__ == "__main__":
    main()

    # Case: 持续检测
    # python -m gtaz.audios.detects_v2

    # Case: 检测 30 秒
    # python -m gtaz.audios.detects_v2 -d 30

    # Case: 匹配到即停止
    # python -m gtaz.audios.detects_v2 -s

    # Case: 自定义阈值和间隔
    # python -m gtaz.audios.detects_v2 -g 0.5 -i 100

    # Case: 强制重新计算模板特征
    # python -m gtaz.audios.detects_v2 -r
