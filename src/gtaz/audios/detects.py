"""GTAV 音频信号检测"""

import time
import threading
from collections import deque
from typing import Optional, Callable, Any
from tclogger import TCLogger, logstr, dict_to_lines

from gtaz.audios.volumes import VolumeRecorder


logger = TCLogger(name="SignalDetector", use_prefix=True, use_prefix_ms=True)


# 阈值比例（相对于窗口最小音量）
GATE_RATIO = 1.8
# 阈值绝对值（触发检测的最小音量）
GATE_VALUE = 10
# 持续时间（毫秒）
DURATION_MS = 800
# 窗口时长（毫秒）
WINDOW_MS = 5000
# 检测冷却时间（毫秒）
COOLDOWN_MS = 2000
# 窗口内第K低音量（用于避免极端低值造成的抖动）
LOWEST_K = 5


class SignalDetector:
    """
    音频信号检测器类。

    检测音量突然增大的事件，基于历史音量窗口和阈值比例。

    算法逻辑：
    1. 获取窗口内的音量数组
    2. 找到数组中第K低音量的最新索引（如果窗口数据量小于K，则跳过检测）
    3. 计算该索引之后超过阈值（kth_lowest_volume * gate_ratio 且 >= gate_value）的样本数
    4. 如果超阈值样本数 >= duration_ms / sample_interval_ms，则视为检测到
    5. 检测成功后进入冷却期，并重置窗口基准音量索引，避免重复检测

    注：使用第K低音量而非最低音量，可以避免极少数极低音量造成的抖动，提高检测稳定性。
    """

    def __init__(
        self,
        recorder: Optional[VolumeRecorder] = None,
        gate_ratio: float = GATE_RATIO,
        gate_value: int = GATE_VALUE,
        duration_ms: float = DURATION_MS,
        window_ms: int = WINDOW_MS,
        detection_callback: Optional[Callable[[dict], Any]] = None,
        cooldown_ms: float = COOLDOWN_MS,
        lowest_k: int = LOWEST_K,
    ):
        """
        初始化信号检测器。

        :param recorder: 音量记录器实例，如果为 None 则创建默认实例
        :param gate_ratio: 阈值比例（相对于第K低音量），默认 1.8 倍
        :param gate_value: 阈值绝对值（触发检测的最小音量），默认 10
        :param duration_ms: 音量突增需要持续的时间（毫秒），默认 500ms
        :param window_ms: 窗口时长（毫秒），默认 5000ms
        :param detection_callback: 检测到信号时的回调函数，接收检测信息字典
        :param cooldown_ms: 检测冷却时间（毫秒），默认 2000ms
        :param lowest_k: 使用窗口内第K低的音量作为基准（避免极端低值抖动），默认 5
        """
        # 如果未传入 recorder，则创建默认实例
        if recorder is None:
            recorder = VolumeRecorder(window_duration_ms=window_ms)

        self.recorder = recorder
        self.gate_ratio = gate_ratio
        self.gate_value = gate_value
        self.duration_ms = duration_ms
        self.window_ms = window_ms
        self.detection_callback = detection_callback
        self.cooldown_ms = cooldown_ms
        self.lowest_k = lowest_k

        # 检测器运行状态
        self._is_running = False
        # 检测统计
        self._detection_count = 0
        # 上次检测时间戳（用于冷却）
        self._last_detection_time: Optional[float] = None
        # 上次检测时的基准音量时间戳（用于重置窗口）
        self._last_base_timestamp: Optional[float] = None
        # 采样间隔（从 monitor 获取）
        self._sample_interval_ms = self.recorder.monitor.sample_interval_ms

    def _get_history_min_volume(self) -> Optional[int]:
        """
        获取历史最小音量（已废弃，保留用于兼容）。

        :返回: 最小音量，如果没有历史数据则返回 None
        """
        return self.recorder.get_history_min()

    def _calculate_gate_volume(self, base_volume: Optional[int]) -> Optional[float]:
        """
        根据基准音量（第K低音量）计算阈值音量。

        :param base_volume: 基准音量（第K低音量）
        :return: 阈值音量，如果无法计算则返回 None
        """
        if base_volume is None:
            return None
        return base_volume * self.gate_ratio

    def _is_in_cooldown(self, current_time: float) -> bool:
        """
        判断是否处于冷却期。

        :param current_time: 当前时间戳（秒）
        :return: 是否处于冷却期
        """
        if self._last_detection_time is None:
            return False
        elapsed_ms = (current_time - self._last_detection_time) * 1000
        return elapsed_ms < self.cooldown_ms

    def _detect(self, volume: int, timestamp: float) -> bool:
        """
        检测给定音量是否触发信号（内部方法）。

        :param volume: 当前音量（0-100）
        :param timestamp: 时间戳（秒）
        :return: 是否检测到信号
        """
        # 检查是否处于冷却期
        if self._is_in_cooldown(timestamp):
            return False

        # 获取窗口内的音量数据（带时间戳）
        window_data = self.recorder.get_window_volumes_with_timestamps()
        if not window_data:
            return False

        # 过滤掉已经使用过的数据（时间戳 <= _last_base_timestamp）
        if self._last_base_timestamp is not None:
            window_data = [
                (ts, vol) for ts, vol in window_data if ts > self._last_base_timestamp
            ]
            if not window_data:
                return False

        # 检查窗口数据量是否足够
        if len(window_data) < self.lowest_k:
            # 数据量不足K个，跳过检测
            return False

        # 获取窗口内第K低的音量和阈值
        volumes = [vol for _, vol in window_data]
        sorted_volumes = sorted(volumes)
        kth_lowest_volume = sorted_volumes[self.lowest_k - 1]  # 第K低（索引从0开始）
        gate_volume = self._calculate_gate_volume(kth_lowest_volume)
        if gate_volume is None:
            return False

        # 找到第K低音量的最新索引（如果有多个相同值，取最后一个）
        base_index = -1
        base_timestamp = None
        for i in range(len(window_data) - 1, -1, -1):
            if window_data[i][1] == kth_lowest_volume:
                base_index = i
                base_timestamp = window_data[i][0]
                break

        if base_index == -1 or base_index >= len(window_data) - 1:
            # 没有找到基准值，或基准值是最后一个，无法检测
            return False

        # 计算基准值索引之后超过阈值的样本数
        # 阈值条件：1) >= gate_volume，2) >= gate_value
        above_gate_count = 0
        for i in range(base_index + 1, len(window_data)):
            vol = window_data[i][1]
            if vol >= gate_volume and vol >= self.gate_value:
                above_gate_count += 1

        # 计算所需的样本数
        required_samples = int(self.duration_ms / self._sample_interval_ms)

        # 判断是否满足条件
        if above_gate_count >= required_samples:
            self._on_detection(
                timestamp,
                kth_lowest_volume,
                gate_volume,
                above_gate_count,
                base_timestamp,
            )
            return True

        return False

    def detect(self) -> bool:
        """
        检测当前音量是否触发信号。

        自动从 recorder.monitor 获取当前音量和时间戳，
        并自动记录到 recorder 中，然后调用 _detect 进行检测。

        :return: 是否检测到信号
        """
        # 获取当前音量
        volume_percent = self.recorder.monitor.get_volume_percent()
        current_time = time.time()
        if volume_percent is None:
            return False
        # 记录音量
        self.recorder.record(volume_percent, current_time)
        # 调用内部检测方法
        return self._detect(volume_percent, current_time)

    def _on_detection(
        self,
        timestamp: float,
        base_volume: int,
        gate_volume: float,
        above_gate_count: int,
        base_timestamp: float,
    ):
        """
        检测到信号时的处理。

        :param timestamp: 检测时间戳（秒）
        :param base_volume: 窗口内基准音量（第K低音量）
        :param gate_volume: 阈值音量
        :param above_gate_count: 超阈值样本数
        :param base_timestamp: 基准值的时间戳
        """
        self._detection_count += 1
        self._last_detection_time = timestamp  # 记录检测时间，用于冷却
        self._last_base_timestamp = base_timestamp  # 记录基准值时间戳，用于重置窗口

        # 获取窗口统计信息
        window_stats = self.recorder.get_window_stats()

        detection_info = {
            "detection_count": self._detection_count,
            "timestamp": timestamp,
            "base_volume": base_volume,
            "gate_volume": gate_volume,
            "gate_ratio": self.gate_ratio,
            "gate_value": self.gate_value,
            "duration_ms": self.duration_ms,
            "lowest_k": self.lowest_k,
            "above_gate_count": above_gate_count,
            "window_stats": window_stats,
        }

        # 输出日志
        self._log_detection_info(detection_info)

        # 调用回调函数
        if self.detection_callback:
            try:
                self.detection_callback(detection_info)
            except Exception as e:
                logger.warn(f"检测回调函数执行出错: {e}")

    def _log_detection_info(self, detection_info: dict):
        """
        输出检测信息日志。

        :param detection_info: 检测信息字典
        """
        print()
        logger.note("=" * 50)
        logger.okay(f"检测到音量突增信号 #{detection_info['detection_count']}")
        logger.note("=" * 50)
        timestamp = detection_info["timestamp"]
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        ms = int((timestamp % 1) * 1000)

        v_min, v_avg, v_max = detection_info["window_stats"]
        ss = logstr.note("/")
        volume_str = f"{v_min}{ss}{v_avg:.1f}{ss}{v_max}"
        info_dict = {
            "触发时间": f"{time_str}.{ms:03d}",
            "基准音量": detection_info["base_volume"],
            # "阈值音量": f"{detection_info['gate_volume']:.1f}",
            # "超阈值样本数": detection_info["above_gate_count"],
            "窗口音量范围": volume_str,
        }
        logger.note(dict_to_lines(info_dict, key_prefix="* "))

        # 恢复显示之前的音量字符
        self.recorder.monitor.log_line_buffer()

    def start(self):
        """
        启动检测器（标记为运行状态）。

        注意：实际的检测逻辑需要在 VolumeMonitor 的回调中调用 detect() 方法。
        """
        if self._is_running:
            logger.warn("检测器已在运行中")
            return
        self._is_running = True
        self._detection_count = 0
        self._last_min_timestamp = None
        logger.okay("检测器已启动")
        info_dict = {
            "阈值比例": f"{self.gate_ratio}x",
            "阈值绝对值": self.gate_value,
            "持续时间": f"{self.duration_ms}ms",
            "窗口时长": f"{self.window_ms}ms",
            "冷却时间": f"{self.cooldown_ms}ms",
            "基准音量": f"第{self.lowest_k}低",
        }
        logger.note(dict_to_lines(info_dict, key_prefix="* "))

    def stop(self):
        """
        停止检测器。
        """
        if not self._is_running:
            logger.warn("检测器未在运行")
            return
        self._is_running = False
        logger.okay("检测器已停止")
        logger.note(f"总检测次数: {self._detection_count}")
        # 重置冷却时间和窗口时间戳
        self._last_detection_time = None
        self._last_base_timestamp = None

    def _log_volume_sample(
        self, sample_count: int, group_volumes: list[int]
    ) -> tuple[int, list[int], Optional[int]]:
        """
        记录并输出单个音量样本的日志。

        :param sample_count: 当前采样计数
        :param group_volumes: 当前组的音量列表
        :return: (更新后的 sample_count, 更新后的 group_volumes, volume_percent 或 None)
        """
        # 检测信号（内部会自动获取音量和记录）
        if self.is_running:
            detected = self.detect()
            if detected:
                return sample_count, group_volumes, None  # 返回 None 表示检测到信号
            # 获取最新记录的音量用于日志显示
            volume_data = self.recorder.get_window_volumes_with_timestamps()
            if volume_data:
                volume_percent = volume_data[-1][1]
            else:
                # 无法获取音量，跳过日志输出
                self.recorder.monitor.sleep_interval()
                return sample_count, group_volumes, -1  # 返回 -1 表示跳过
        else:
            # 检测器未运行，直接获取音量
            volume_percent = self.recorder.monitor.get_volume_percent()
            if volume_percent is None:
                self.recorder.monitor.sleep_interval()
                return sample_count, group_volumes, -1  # 返回 -1 表示跳过
        # 输出日志
        group_volumes.append(volume_percent)
        volume_char = self.recorder.monitor.get_volume_char(volume_percent)
        self.recorder.monitor._log_volume_char(volume_char, sample_count)
        if self.recorder.monitor._is_last_in_group(sample_count):
            self.recorder.monitor._log_group_stats(group_volumes)
            group_volumes = []
        sample_count += 1
        self.recorder.monitor.sleep_interval()

        return sample_count, group_volumes, volume_percent

    def stop_after_detect(self, count: int = 1, interval: float = 0) -> int:
        """
        持续检测音量信号，检测到指定次数后等待并退出。

        自动启动检测器，循环调用 detect() 方法并输出音量日志，
        检测到 count 次信号后等待 interval 秒，然后停止检测器并退出。

        :param count: 检测到多少次信号后退出，默认 1 次
        :param interval: 检测到 count 次信号后等待多少秒再退出，默认 0 秒（立即退出）
        :return: 实际检测到的次数
        """
        # 启动检测器
        if not self._is_running:
            self.start()
        logger.note(f"开始检测音量信号，检测到 {count} 次后等待 {interval} 秒退出...")
        sample_count = 0
        group_volumes = []
        detected_count = 0
        try:
            while detected_count < count:
                # 记录并输出音量样本日志
                sample_count, group_volumes, result = self._log_volume_sample(
                    sample_count, group_volumes
                )
                # 检查结果
                if result is None:
                    # 检测到信号
                    detected_count += 1
                    print()
                    logger.note(f"已触发信号次数: {detected_count}/{count}")
                    self.recorder.monitor.log_line_buffer()
                    if detected_count >= count:
                        break
                elif result == -1:
                    # 跳过此次采样
                    continue
        except KeyboardInterrupt:
            logger.note("检测被用户中断")
        finally:
            # 输出最后一组统计
            if sample_count % self.recorder.monitor.samples_per_group != 0:
                self.recorder.monitor._log_group_stats(group_volumes)
            # 等待指定时间
            if interval > 0 and detected_count >= count:
                logger.note(f"等待 {interval} 秒...")
                time.sleep(interval)
            # 停止检测器
            self.stop()

        return detected_count

    @property
    def is_running(self) -> bool:
        """获取检测器运行状态。"""
        return self._is_running

    def get_detection_count(self) -> int:
        """获取检测次数。"""
        return self._detection_count

    def reset_detection_count(self):
        """重置检测次数。"""
        self._detection_count = 0

    def __repr__(self) -> str:
        min_volume = self._get_history_min_volume()
        gate_volume = self._calculate_gate_volume(min_volume)
        gate_str = f"{gate_volume:.1f}" if gate_volume is not None else "None"
        return (
            f"SignalDetector("
            f"gate_ratio={self.gate_ratio}, "
            f"gate_value={self.gate_value}, "
            f"duration_ms={self.duration_ms}, "
            f"window_ms={self.window_ms}, "
            f"cooldown_ms={self.cooldown_ms}, "
            f"lowest_k={self.lowest_k}, "
            f"is_running={self._is_running}, "
            f"detection_count={self._detection_count}, "
            f"min_volume={min_volume}, "
            f"gate_volume={gate_str})"
        )


def test_signal_detector():
    """测试信号检测器。"""
    # 创建音量记录器（会自动创建 monitor，5 秒窗口）
    recorder = VolumeRecorder(window_duration_ms=WINDOW_MS)
    logger.note(f"音量记录器信息: {recorder}")

    # 创建信号检测器
    detector = SignalDetector(
        recorder=recorder,
        gate_ratio=GATE_RATIO,
        duration_ms=DURATION_MS,
    )
    logger.note(f"信号检测器信息: {detector}")

    # 可选：检查游戏是否运行
    recorder.monitor.find_audio_session()
    logger.note(f"音量监控器信息: {recorder.monitor}")

    # 启动检测器
    detector.start()

    try:
        # 持续监控，同时进行检测
        logger.note("开始监控音量并检测突增信号...")
        logger.note("按 Ctrl+C 停止监控")

        # 使用记录器的监控方法，在记录后进行检测
        sample_count = 0
        group_volumes = []

        while True:
            # 检测信号（内部会自动获取音量和记录）
            if detector.is_running:
                detected = detector.detect()
                if not detected:
                    # 如果返回 False 可能是因为无法获取音量
                    volume_percent = recorder.monitor.get_volume_percent()
                    if volume_percent is None:
                        logger.warn("无法获取音量，跳过此次采样")
                        recorder.monitor.sleep_interval()
                        continue
                else:
                    # 检测成功，获取最新记录的音量用于日志显示
                    volume_data = recorder.get_window_volumes_with_timestamps()
                    if volume_data:
                        volume_percent = volume_data[-1][1]
                    else:
                        volume_percent = 0
            else:
                volume_percent = recorder.monitor.get_volume_percent()
                if volume_percent is None:
                    logger.warn("无法获取音量，跳过此次采样")
                    recorder.monitor.sleep_interval()
                    continue

            # 输出日志
            group_volumes.append(volume_percent)
            volume_char = recorder.monitor.get_volume_char(volume_percent)
            recorder.monitor._log_volume_char(volume_char, sample_count)
            if recorder.monitor._is_last_in_group(sample_count):
                recorder.monitor._log_group_stats(group_volumes)
                group_volumes = []

            sample_count += 1
            recorder.monitor.sleep_interval()

    except KeyboardInterrupt:
        if sample_count % recorder.monitor.samples_per_group != 0:
            recorder.monitor._log_group_stats(group_volumes)
        logger.note("监控已停止")
    finally:
        # 停止检测器
        detector.stop()


def test_stop_after_detect():
    """测试 stop_after_detect 方法。"""
    # 创建信号检测器（会自动创建 recorder）
    detector = SignalDetector(
        gate_ratio=GATE_RATIO,
        duration_ms=DURATION_MS,
    )
    logger.note(f"信号检测器信息: {detector}")
    logger.note(f"音量记录器信息: {detector.recorder}")

    # 可选：检查游戏是否运行
    detector.recorder.monitor.find_audio_session()
    logger.note(f"音量监控器信息: {detector.recorder.monitor}")

    # 使用 stop_after_detect 方法
    logger.note("=" * 50)
    logger.note("测试 stop_after_detect 方法")
    logger.note("=" * 50)
    logger.note("按 Ctrl+C 可以中断检测")

    # 检测到 1 次信号后立即退出
    detected_count = detector.stop_after_detect(count=2, interval=0)

    logger.note("=" * 50)
    logger.okay(f"检测完成，共检测到 {detected_count} 次信号")
    logger.note("=" * 50)


if __name__ == "__main__":
    # test_signal_detector()
    test_stop_after_detect()

    # python -m gtaz.audios.detects
