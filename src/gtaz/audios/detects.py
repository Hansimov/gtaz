"""GTAV 音频信号检测"""

import time
import threading
from collections import deque
from typing import Optional, Callable, Any
from tclogger import TCLogger, logstr

from gtaz.audios.volumes import VolumeRecorder, SAMPLE_INTERVAL_MS


logger = TCLogger(name="SignalDetector", use_prefix=True, use_prefix_ms=True)


# 阈值比例（相对于历史最小音量）
GATE_RATIO = 1.8
# 阈值绝对值（触发检测的最小音量）
GATE_VALUE = 10
# 持续时间（毫秒）
DURATION_MS = 500
# 历史窗口（毫秒）
HISTORY_WINDOW_MS = 5000
# 检测冷却时间（毫秒）
COOLDOWN_MS = 1000


class SignalDetector:
    """
    音频信号检测器类。

    检测音量突然增大的事件，基于历史音量窗口和阈值比例。

    算法逻辑：
    1. 获取窗口内的音量数组
    2. 找到数组中最低音量的最新索引（如果有多个最小值，取最后一个）
    3. 计算该索引之后超过阈值（min_volume * gate_ratio 且 >= gate_value）的样本数
    4. 如果超阈值样本数 >= duration_ms / sample_interval_ms，则视为检测到
    5. 检测成功后进入冷却期，并重置窗口最小值索引，避免重复检测
    """

    def __init__(
        self,
        recorder: VolumeRecorder,
        gate_ratio: float = GATE_RATIO,
        gate_value: int = GATE_VALUE,
        duration_ms: float = DURATION_MS,
        history_window_ms: int = HISTORY_WINDOW_MS,
        detection_callback: Optional[Callable[[dict], Any]] = None,
        cooldown_ms: float = COOLDOWN_MS,
    ):
        """
        初始化信号检测器。

        :param recorder: 音量记录器实例
        :param gate_ratio: 阈倿比例（相对于历史最小音量），默认 1.8 倍
        :param gate_value: 阈值绝对值（触发检测的最小音量），默认 10
        :param duration_ms: 音量突增需要持续的时间（毫秒），默认 800ms
        :param history_window_ms: 历史窗口长度（毫秒），默认 5000ms
        :param detection_callback: 检测到信号时的回调函数，接收检测信息字典
        :param cooldown_ms: 检测冷却时间（毫秒），默认 1000ms
        """
        self.recorder = recorder
        self.gate_ratio = gate_ratio
        self.gate_value = gate_value
        self.duration_ms = duration_ms
        self.history_window_ms = history_window_ms
        self.detection_callback = detection_callback
        self.cooldown_ms = cooldown_ms

        # 检测器运行状态
        self._is_running = False
        # 检测统计
        self._detection_count = 0
        # 上次检测时间戳（用于冷却）
        self._last_detection_time: Optional[float] = None
        # 上次检测时的最小值时间戳（用于重置窗口）
        self._last_min_timestamp: Optional[float] = None
        # 采样间隔（从 monitor 获取）
        self._sample_interval_ms = self.recorder.monitor.sample_interval_ms

    def _get_history_min_volume(self) -> Optional[int]:
        """
        获取历史最小音量。

        :返回: 最小音量，如果没有历史数据则返回 None
        """
        return self.recorder.get_history_min()

    def _calculate_gate_volume(self, min_volume: Optional[int]) -> Optional[float]:
        """
        根据历史最小音量计算阈值音量。

        :param min_volume: 历史最小音量
        :return: 阈值音量，如果无法计算则返回 None
        """
        if min_volume is None:
            return None
        return min_volume * self.gate_ratio

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

    def detect(self, volume: int, timestamp: float) -> bool:
        """
        检测给定音量是否触发信号。

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

        # 过滤掉已经使用过的数据（时间戳 <= _last_min_timestamp）
        if self._last_min_timestamp is not None:
            window_data = [
                (ts, vol) for ts, vol in window_data if ts > self._last_min_timestamp
            ]
            if not window_data:
                return False

        # 获取窗口最小音量和阈值
        volumes = [vol for _, vol in window_data]
        min_volume = min(volumes)
        gate_volume = self._calculate_gate_volume(min_volume)
        if gate_volume is None:
            return False

        # 找到最低音量的最新索引（如果有多个，取最后一个）
        min_index = -1
        min_timestamp = None
        for i in range(len(window_data) - 1, -1, -1):
            if window_data[i][1] == min_volume:
                min_index = i
                min_timestamp = window_data[i][0]
                break

        if min_index == -1 or min_index >= len(window_data) - 1:
            # 没有找到最小值，或最小值是最后一个，无法检测
            return False

        # 计算最小值索引之后超过阈值的样本数
        # 阈值条件：1) >= gate_volume，2) >= gate_value
        above_gate_count = 0
        for i in range(min_index + 1, len(window_data)):
            vol = window_data[i][1]
            if vol >= gate_volume and vol >= self.gate_value:
                above_gate_count += 1

        # 计算所需的样本数
        required_samples = int(self.duration_ms / self._sample_interval_ms)

        # 判断是否满足条件
        if above_gate_count >= required_samples:
            self._on_detection(
                timestamp, min_volume, gate_volume, above_gate_count, min_timestamp
            )
            return True

        return False

    def _on_detection(
        self,
        timestamp: float,
        min_volume: int,
        gate_volume: float,
        above_gate_count: int,
        min_timestamp: float,
    ):
        """
        检测到信号时的处理。

        :param timestamp: 检测时间戳（秒）
        :param min_volume: 窗口内最小音量
        :param gate_volume: 阈值音量
        :param above_gate_count: 超阈值样本数
        :param min_timestamp: 最小值的时间戳
        """
        self._detection_count += 1
        self._last_detection_time = timestamp  # 记录检测时间，用于冷却
        self._last_min_timestamp = min_timestamp  # 记录最小值时间戳，用于重置窗口

        # 获取窗口统计信息
        window_stats = self.recorder.get_window_stats()

        detection_info = {
            "detection_count": self._detection_count,
            "timestamp": timestamp,
            "min_volume": min_volume,
            "gate_volume": gate_volume,
            "gate_ratio": self.gate_ratio,
            "gate_value": self.gate_value,
            "duration_ms": self.duration_ms,
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
        logger.note(f"检测到音量突增信号 #{detection_info['detection_count']}")
        logger.note("=" * 50)
        timestamp = detection_info["timestamp"]
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        ms = int((timestamp % 1) * 1000)
        logger.note(f"检测时间: {time_str}.{ms:03d}")
        logger.note(f"窗口内最小音量: {detection_info['min_volume']}")
        logger.note(
            f"阈值音量: {detection_info['gate_volume']:.1f} "
            f"({detection_info['gate_ratio']}x, min={detection_info['gate_value']})"
        )
        logger.note(f"持续时间: {detection_info['duration_ms']}ms")
        logger.note(f"超阈值样本数: {detection_info['above_gate_count']}")

        window_stats = detection_info["window_stats"]
        if window_stats[0] is not None:
            window_avg = (
                f"{window_stats[1]:.1f}" if window_stats[1] is not None else "None"
            )
            logger.note(
                f"窗口统计: "
                f"min={window_stats[0]}, "
                f"avg={window_avg}, "
                f"max={window_stats[2]}"
            )

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
        logger.note(f"阈值比例: {self.gate_ratio}x")
        logger.note(f"阈值绝对值: {self.gate_value}")
        logger.note(f"持续时间: {self.duration_ms}ms")
        logger.note(f"历史窗口: {self.history_window_ms}ms")
        logger.note(f"冷却时间: {self.cooldown_ms}ms")

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
        self._last_min_timestamp = None

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
            f"history_window_ms={self.history_window_ms}, "
            f"cooldown_ms={self.cooldown_ms}, "
            f"is_running={self._is_running}, "
            f"detection_count={self._detection_count}, "
            f"min_volume={min_volume}, "
            f"gate_volume={gate_str})"
        )


def test_signal_detector():
    """测试信号检测器。"""
    # 创建音量记录器（会自动创建 monitor，5 秒历史窗口）
    recorder = VolumeRecorder(window_duration_ms=HISTORY_WINDOW_MS)
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
            volume_percent = recorder.monitor.get_volume_percent()
            current_time = time.time()

            if volume_percent is None:
                logger.warn("无法获取音量，跳过此次采样")
                recorder.monitor.sleep_interval()
                continue

            # 记录音量
            recorder.record(volume_percent, current_time)
            # 检测信号
            if detector.is_running:
                detector.detect(volume_percent, current_time)

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


if __name__ == "__main__":
    test_signal_detector()

    # python -m gtaz.audios.detects
