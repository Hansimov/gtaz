"""实时音频音量检测模块

基于音量阈值的简单检测方案，不使用特征匹配。
当窗口内平均音量和最大音量同时超过阈值时，视为检测到信号。
"""

import argparse
import time

from typing import Callable, Any
from tclogger import TCLogger, logstr

from .sounds import (
    SoundRecorder,
    AUDIO_DEVICE_NAME,
    SAMPLE_RATE,
    CHANNELS,
    WINDOW_MS,
    SAMPLES_PER_GROUP,
    VOLUME_SAMPLE_INTERVAL_MS,
)

logger = TCLogger(
    name="VolumeDetector",
    use_prefix=True,
    use_prefix_ms=True,
)

# 检测间隔（毫秒）
DETECT_INTERVAL_MS = 200

# 平均音量阈值（音量百分比）
AVG_VOLUME_THRESHOLD = 25
# 最大音量阈值（音量百分比）
MAX_VOLUME_THRESHOLD = 45
# 最小预热样本数，避免开始时的抖动
MIN_WARMUP_SAMPLES = 10


class VolumeDetector:
    """基于音量阈值的实时检测器

    检测逻辑：当窗口内平均音量 >= AVG_VOLUME_THRESHOLD 且
    最大音量 >= MAX_VOLUME_THRESHOLD 时，视为检测到信号。
    """

    def __init__(
        self,
        device_name: str = AUDIO_DEVICE_NAME,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        window_ms: int = WINDOW_MS,
        avg_threshold: int = AVG_VOLUME_THRESHOLD,
        max_threshold: int = MAX_VOLUME_THRESHOLD,
        detect_interval_ms: int = DETECT_INTERVAL_MS,
    ):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_ms = window_ms
        self.avg_threshold = avg_threshold
        self.max_threshold = max_threshold
        self.detect_interval_ms = detect_interval_ms

        # 音频录制器
        self.recorder = SoundRecorder(
            device_name=device_name,
            sample_rate=sample_rate,
            channels=channels,
            window_ms=window_ms,
        )

        # 最后检测时间
        self._last_detect_time: float = 0

        # 音量显示相关
        self._sample_count = 0
        self._group_volumes: list[int] = []
        self._line_buffer: list[str] = []
        self._current_tick_detected = False

        self._log_init()

    def _log_init(self):
        """初始化日志"""
        logger.okay(
            f"音量检测器初始化完成 "
            f"(平均阈值: {self.avg_threshold}, 最大阈值: {self.max_threshold})"
        )

    def _is_first_in_group(self) -> bool:
        """判断是否是组内第一个。"""
        return self._sample_count % SAMPLES_PER_GROUP == 0

    def _is_last_in_group(self) -> bool:
        """判断是否是组内最后一个。"""
        return (self._sample_count + 1) % SAMPLES_PER_GROUP == 0

    def _log_volume_char(self, volume_char: str, highlight: bool = False):
        """输出音量字符。"""
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

    def _log_group_stats(self):
        """输出一组音量的统计信息。"""
        if self._group_volumes:
            min_vol, avg_vol, max_vol = self._calculate_stats(self._group_volumes)
            vol_strs = [
                logstr.mesg(f"{round(v):2d}") for v in [min_vol, avg_vol, max_vol]
            ]
            vol_line = "/".join(vol_strs)
            # 显示是否达到阈值
            avg_ok = avg_vol >= self.avg_threshold
            max_ok = max_vol >= self.max_threshold
            if avg_ok and max_ok:
                status_str = logstr.okay("✓")
            else:
                status_str = logstr.file("·")
            logger.note(f" [{vol_line}] {status_str}", use_prefix=False)
        else:
            logger.note("", use_prefix=False)

    def detect(self) -> tuple[bool, int, int]:
        """执行一次检测

        :return: (是否检测到, 平均音量, 最大音量)
        """
        # 启动后的预热期间不检测，等待缓冲区填充
        if self._sample_count < MIN_WARMUP_SAMPLES:
            return False, 0, 0

        if not self._group_volumes:
            return False, 0, 0

        min_vol, avg_vol, max_vol = self._calculate_stats(self._group_volumes)
        detected = avg_vol >= self.avg_threshold and max_vol >= self.max_threshold
        return detected, int(avg_vol), int(max_vol)

    def detect_loop(self, duration: float = None):
        """持续检测循环

        :param duration: 检测时长（秒），None 表示持续检测
        """
        # 启动音频流
        if not self.recorder.start_stream():
            logger.warn("无法启动音频流")
            return

        logger.note(
            f"开始音量检测 (平均阈值: {self.avg_threshold}, 最大阈值: {self.max_threshold})"
        )
        if duration:
            logger.note(f"检测时长: {duration}秒")

        start_time = time.time()
        self._sample_count = 0
        self._group_volumes = []
        self._current_tick_detected = False

        try:
            while True:
                # 检查时长
                if duration and (time.time() - start_time) >= duration:
                    if self._sample_count % SAMPLES_PER_GROUP != 0:
                        self._log_group_stats()
                    logger.note("检测时长已到")
                    break

                # 重置当前 tick 的检测状态
                self._current_tick_detected = False

                # 检测间隔到达时执行检测
                current_time = time.time()
                if (
                    current_time - self._last_detect_time
                ) * 1000 >= self.detect_interval_ms:
                    self._last_detect_time = current_time
                    detected, avg_vol, max_vol = self.detect()
                    if detected:
                        self._current_tick_detected = True

                # 获取当前音量并显示
                volume_percent = self.recorder.get_volume_percent()
                self._group_volumes.append(volume_percent)
                volume_char = self.recorder.get_volume_char(volume_percent)
                self._log_volume_char(
                    volume_char, highlight=self._current_tick_detected
                )

                # 组结束时输出统计
                if self._is_last_in_group():
                    self._log_group_stats()
                    self._group_volumes = []

                self._sample_count += 1
                time.sleep(VOLUME_SAMPLE_INTERVAL_MS / 1000)

        except KeyboardInterrupt:
            if self._sample_count % SAMPLES_PER_GROUP != 0:
                self._log_group_stats()
            logger.note("\n检测到 Ctrl+C，正在退出...")

        finally:
            self.recorder.stop_stream()

    def detect_then_stop(
        self,
        timeout: float = None,
        on_match: Callable[[int, int], Any] = None,
    ) -> tuple[bool, int, int]:
        """检测直到匹配成功或超时

        :param timeout: 超时时间（秒），None 表示无超时
        :param on_match: 匹配成功时的回调函数，接收 (avg_vol, max_vol) 参数
        :return: (是否匹配成功, 平均音量, 最大音量)
        """
        # 启动音频流
        if not self.recorder.start_stream():
            logger.warn("无法启动音频流")
            return False, 0, 0

        logger.note(
            f"开始检测 (平均阈值: {self.avg_threshold}, 最大阈值: {self.max_threshold})，匹配到即停止"
        )
        if timeout:
            logger.note(f"超时时间: {timeout}秒")

        start_time = time.time()
        self._sample_count = 0
        self._group_volumes = []
        self._current_tick_detected = False
        matched = False
        final_avg = 0
        final_max = 0

        try:
            while True:
                # 检查超时
                if timeout and (time.time() - start_time) >= timeout:
                    if self._sample_count % SAMPLES_PER_GROUP != 0:
                        self._log_group_stats()
                    logger.note("检测超时")
                    break

                # 重置当前 tick 的检测状态
                self._current_tick_detected = False

                # 检测间隔到达时执行检测
                current_time = time.time()
                if (
                    current_time - self._last_detect_time
                ) * 1000 >= self.detect_interval_ms:
                    self._last_detect_time = current_time
                    detected, avg_vol, max_vol = self.detect()

                    if detected:
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
                            f"[{elapsed:.1f}s] 检测到信号! "
                            f"平均={logstr.okay(f'{avg_vol}')} 最大={logstr.okay(f'{max_vol}')}"
                        )
                        matched = True
                        final_avg = avg_vol
                        final_max = max_vol

                        # 调用回调
                        if on_match:
                            on_match(avg_vol, max_vol)

                        break

                # 获取当前音量并显示
                volume_percent = self.recorder.get_volume_percent()
                self._group_volumes.append(volume_percent)
                volume_char = self.recorder.get_volume_char(volume_percent)
                self._log_volume_char(
                    volume_char, highlight=self._current_tick_detected
                )

                # 组结束时输出统计
                if self._is_last_in_group():
                    self._log_group_stats()
                    self._group_volumes = []

                self._sample_count += 1
                time.sleep(VOLUME_SAMPLE_INTERVAL_MS / 1000)

        except KeyboardInterrupt:
            if self._sample_count % SAMPLES_PER_GROUP != 0:
                self._log_group_stats()
            logger.note("\n检测到 Ctrl+C，正在退出...")

        finally:
            self.recorder.stop_stream()

        return matched, final_avg, final_max


class VolumeDetectorArgParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="实时音量阈值检测")
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
            "-a",
            "--avg-threshold",
            type=int,
            default=AVG_VOLUME_THRESHOLD,
            help=f"平均音量阈值（默认: {AVG_VOLUME_THRESHOLD}）",
        )
        self.parser.add_argument(
            "-m",
            "--max-threshold",
            type=int,
            default=MAX_VOLUME_THRESHOLD,
            help=f"最大音量阈值（默认: {MAX_VOLUME_THRESHOLD}）",
        )
        self.parser.add_argument(
            "-n",
            "--device-name",
            type=str,
            default=AUDIO_DEVICE_NAME,
            help=f"音频设备名称（默认: {AUDIO_DEVICE_NAME}）",
        )
        self.parser.add_argument(
            "-s",
            "--stop-on-match",
            action="store_true",
            help="匹配到即停止",
        )

    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()


def main():
    """命令行入口"""
    args = VolumeDetectorArgParser().parse()

    # 创建检测器
    detector = VolumeDetector(
        device_name=args.device_name,
        avg_threshold=args.avg_threshold,
        max_threshold=args.max_threshold,
    )

    # 运行检测
    if args.stop_on_match:
        matched, avg_vol, max_vol = detector.detect_then_stop(timeout=args.duration)
        if matched:
            logger.okay(f"检测完成，平均音量: {avg_vol}, 最大音量: {max_vol}")
        else:
            logger.note("未检测到目标信号")
    else:
        detector.detect_loop(duration=args.duration)


if __name__ == "__main__":
    main()

    # Case: 持续检测
    # python -m gtaz.audios.detects_v3

    # Case: 检测 30 秒
    # python -m gtaz.audios.detects_v3 -d 30

    # Case: 匹配到即停止
    # python -m gtaz.audios.detects_v3 -s

    # Case: 自定义阈值
    # python -m gtaz.audios.detects_v3 -a 30 -m 50
