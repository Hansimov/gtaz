"""GTAV 音量监控"""

"""
References:
- https://github.com/AndreMiras/pycaw
- https://docs.microsoft.com/en-us/windows/win32/coreaudio/core-audio-interfaces

安装依赖：

```sh
pip install pycaw
```
"""

import time
import warnings
from collections import deque
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
from typing import Optional
from tclogger import TCLogger, logstr


# 抑制 pycaw 的 COMError 警告
warnings.filterwarnings("ignore", category=UserWarning, module="pycaw.utils")

logger = TCLogger(name="VolumeMonitor", use_prefix=True, use_prefix_ms=True)


# GTAV 增强版进程名
GTAV_PROCESS_NAME = "GTA5_Enhanced.exe"
# 默认音频输出设备名称
GTAV_AUDIO_DEVICE_NAME = "CABLE Input"
# 音量字符
VOLUME_CHARS = "▁▂▃▅▆▇"
# 音量量化位数
VOLUME_BITS = len(VOLUME_CHARS)
# 采样间隔（毫秒）
SAMPLE_INTERVAL_MS = 100
# 每组输出的样本数
SAMPLES_PER_GROUP = 25
# 默认音量增益系数（幂函数的指数，<1 增强低音量区分度）
VOLUME_GAIN = 0.5


def sleep_ms(milliseconds: int):
    """
    按毫秒睡眠。
    :param milliseconds: 毫秒数
    """
    time.sleep(milliseconds / 1000)


class AudioDeviceSession:
    """
    音频设备会话类。

    负责音频设备的查找、连接和音频计量接口的管理。
    """

    def __init__(self, device_name: str = GTAV_AUDIO_DEVICE_NAME):
        """
        初始化音频设备会话。

        :param device_name: 音频设备名称（包含此关键字的设备）
        """
        self.device_name = device_name
        self._device = None
        self._meter: Optional[IAudioMeterInformation] = None

    def setup(self) -> bool:
        """
        设置音频设备和计量接口。

        :return: 是否成功设置
        """
        try:
            # 遍历所有设备，查找匹配的设备名称
            devices = AudioUtilities.GetAllDevices()
            for device in devices:
                try:
                    device_name = device.FriendlyName
                    # 检查设备名称是否包含目标关键字（不区分大小写）
                    if self.device_name.lower() in device_name.lower():
                        self._device = device
                        self._setup_meter_from_device(device)
                        logger.okay(f"音频输出设备: {device_name}")
                        return True
                except Exception as e:
                    continue

            # 未找到匹配设备，使用系统默认设备
            logger.warn(f"未找到包含 '{self.device_name}' 的设备，使用系统默认设备")
            device = AudioUtilities.GetSpeakers()
            self._device = device
            self._setup_meter_from_device(device)
            logger.okay(f"音频输出设备: {device.FriendlyName}")
            return True

        except Exception as e:
            logger.err(f"无法设置音频设备: {e}")
            return False

    def _setup_meter_from_device(self, device):
        """从设备获取音频计量接口。"""
        try:
            if hasattr(device, "_dev"):
                imm_device = device._dev
            elif hasattr(device, "QueryInterface"):
                imm_device = device
            else:
                imm_device = device
            interface = imm_device.Activate(
                IAudioMeterInformation._iid_, CLSCTX_ALL, None
            )
            self._meter = interface.QueryInterface(IAudioMeterInformation)
            logger.okay("成功获取音频计量接口")
        except Exception as e:
            logger.warn(f"获取音频计量接口失败: {e}")
            self._meter = None

    @property
    def meter(self) -> Optional[IAudioMeterInformation]:
        """获取音频计量接口。"""
        return self._meter

    @property
    def device(self):
        """获取音频设备。"""
        return self._device

    def get_peak_volume(self) -> Optional[float]:
        """
        获取当前峰值音量（0.0-1.0）。
        :return: 峰值音量，获取失败则返回 None
        """
        try:
            if self.meter:
                return self.meter.GetPeakValue()
            return None
        except Exception as e:
            logger.warn(f"获取峰值音量时出错: {e}")
            return None

    def __repr__(self) -> str:
        device_name = self._device.FriendlyName if self._device else None
        has_meter = self._meter is not None
        return (
            f"AudioDeviceSession("
            f"device_name={self.device_name}, "
            f"actual_device={device_name}, "
            f"has_meter={has_meter})"
        )


class VolumeMonitor:
    """
    GTAV 音量监控类。

    使用 AudioDeviceSession 进行音量监控，支持音量增益和实时数据回调。
    """

    def __init__(
        self,
        device_session: Optional[AudioDeviceSession] = None,
        process_name: str = GTAV_PROCESS_NAME,
        sample_interval_ms: int = SAMPLE_INTERVAL_MS,
        samples_per_group: int = SAMPLES_PER_GROUP,
        volume_gain: float = VOLUME_GAIN,
    ):
        """
        初始化音量监控器。

        :param device_session: 音频设备会话（如果为 None 则自动创建）
        :param process_name: 进程名称（用于验证游戏运行）
        :param sample_interval_ms: 采样间隔（毫秒）
        :param samples_per_group: 每组输出的样本数
        :param volume_gain: 音量增益系数（幂函数指数，<1 增强低音量区分度）
        """
        # 如果未传入 device_session，自动创建默认实例
        if device_session is None:
            device_session = AudioDeviceSession(device_name=GTAV_AUDIO_DEVICE_NAME)
            if not device_session.setup():
                logger.warn("无法设置音频设备，VolumeMonitor 可能无法正常工作")

        self.device_session = device_session
        self.process_name = process_name
        self.sample_interval_ms = sample_interval_ms
        self.samples_per_group = samples_per_group
        self.volume_gain = volume_gain
        self._audio_session = None
        self._is_running = False

    def find_audio_session(self):
        """
        查找 GTAV 进程的音频会话（用于验证游戏运行）。

        :return: 音频会话对象，未找到则返回 None
        """
        try:
            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                if session.Process and session.Process.name() == self.process_name:
                    self._audio_session = session
                    logger.okay(f"已找到 GTAV 音频会话: PID={session.Process.pid}")
                    return session
            logger.warn(f"未找到 GTAV 音频会话 (进程名: {self.process_name})")
            return None
        except Exception as e:
            logger.warn(f"查找音频会话时出错: {e}")
            return None

    @property
    def audio_session(self):
        """获取缓存的音频会话，如果未缓存则重新查找。"""
        if self._audio_session is None:
            self.find_audio_session()
        return self._audio_session

    def get_peak_volume(self) -> Optional[float]:
        """
        获取当前峰值音量（0.0-1.0）。

        :return: 峰值音量，获取失败则返回 None
        """
        if not self.device_session:
            return None
        return self.device_session.get_peak_volume()

    def apply_volume_gain(self, volume: float) -> float:
        """
        应用音量增益。
        使用幂函数进行增益：output = volume ^ gain
        - gain < 1: 增强低音量区分度（推荐 0.3-0.5）
        - gain = 1: 线性，不变
        - gain > 1: 压缩低音量，扩展高音量
        :param volume: 原始音量（0.0-1.0）
        :return: 增益后的音量（0.0-1.0）
        """
        if volume is None or volume < 0:
            return 0.0
        return min(1.0, volume**self.volume_gain)

    def get_volume_percent(self, apply_gain: bool = True) -> Optional[int]:
        """
        获取当前音量百分比（0-100）。
        :param apply_gain: 是否应用音量增益
        :return: 音量百分比，获取失败则返回 None
        """
        peak = self.get_peak_volume()
        if peak is not None:
            if apply_gain:
                peak = self.apply_volume_gain(peak)
            return int(peak * 100)
        return None

    def sleep_interval(self):
        """按采样间隔睡眠。"""
        sleep_ms(self.sample_interval_ms)

    @staticmethod
    def _calculate_stats(volumes: list[int]) -> tuple[int, float, int]:
        """
        计算音量统计信息。

        :param volumes: 音量数据列表
        :return: (min, avg, max)
        """
        if not volumes:
            return 0, 0.0, 0
        return min(volumes), sum(volumes) / len(volumes), max(volumes)

    def _is_first_in_group(self, sample_count: int) -> bool:
        """
        判断是否是组内第一个。
        :param sample_count: 当前采样计数
        :return: 是否是组内第一个
        """
        return sample_count % self.samples_per_group == 0

    def _is_last_in_group(self, sample_count: int) -> bool:
        """
        判断是否是组内最后一个。
        :param sample_count: 当前采样计数
        :return: 是否是组内最后一个
        """
        return (sample_count + 1) % self.samples_per_group == 0

    @staticmethod
    def get_volume_char(volume_percent: int) -> str:
        """
        将音量百分比映射到音量字符。

        :param volume_percent: 音量百分比（0-100）
        :return: 对应的音量字符
        """
        # 将 0-100 映射到 0-(VOLUME_BITS-1) 的索引
        index = min(VOLUME_BITS - 1, int(volume_percent / 100 * VOLUME_BITS))
        return VOLUME_CHARS[index]

    def _log_volume_char(self, volume_char: str, sample_count: int):
        """
        输出音量字符。
        :param volume_char: 音量字符
        :param sample_count: 当前采样计数
        """
        if self._is_first_in_group(sample_count):
            logger.mesg(volume_char, end="")
        elif self._is_last_in_group(sample_count):
            logger.mesg(volume_char, use_prefix=False, end="")
        else:
            logger.mesg(volume_char, use_prefix=False, end="")

    @staticmethod
    def _log_group_stats(volumes: list[int]):
        """
        输出一组音量的统计信息。

        :param volumes: 音量数据列表
        """
        if volumes:
            min_vol, avg_vol, max_vol = VolumeMonitor._calculate_stats(volumes)
            vol_strs = [
                logstr.mesg(f"{round(v):2d}") for v in [min_vol, avg_vol, max_vol]
            ]
            vol_line = "/".join(vol_strs)
            logger.note(f" [{vol_line}]", use_prefix=False)
        else:
            logger.note("", use_prefix=False)  # 只换行

    def _log_monitor_start(self):
        """输出监控开始的信息。"""
        logger.note(f"开始监控 GTAV 音量 (采样间隔: {self.sample_interval_ms}ms)")
        logger.note(
            f"每组 {self.samples_per_group} 个样本 "
            f"({self.samples_per_group * self.sample_interval_ms / 1000:.1f} 秒)"
        )
        logger.note(f"音量增益: {self.volume_gain} (幂函数指数)")

    def start(self):
        """
        开始监控音量并输出到日志。
        """
        if not self.device_session or not self.device_session.meter:
            logger.fail("无法开始监控: 音频设备会话未设置")
            return

        if self._is_running:
            logger.warn("监控器已在运行中")
            return

        self._is_running = True
        self._log_monitor_start()
        sample_count = 0
        group_volumes = []

        try:
            while self._is_running:
                volume_percent = self.get_volume_percent()
                if volume_percent is None:
                    logger.warn("无法获取音量，跳过此次采样")
                    self.sleep_interval()
                    continue
                group_volumes.append(volume_percent)
                volume_char = self.get_volume_char(volume_percent)
                self._log_volume_char(volume_char, sample_count)
                if self._is_last_in_group(sample_count):
                    self._log_group_stats(group_volumes)
                    group_volumes = []
                sample_count += 1
                self.sleep_interval()
        except KeyboardInterrupt:
            if sample_count % self.samples_per_group != 0:
                self._log_group_stats(group_volumes)
            logger.note("监控已停止")
        except Exception as e:
            logger.warn(f"监控过程中出错: {e}")
        finally:
            self._is_running = False

    def stop(self):
        """
        停止监控。
        """
        if not self._is_running:
            logger.warn("监控器未在运行")
            return
        self._is_running = False
        logger.okay("监控器已停止")

    @property
    def is_running(self) -> bool:
        """获取监控器运行状态。"""
        return self._is_running

    def refresh(self):
        """刷新音频会话缓存。"""
        self._audio_session = None
        return self.audio_session

    def __repr__(self) -> str:
        has_session = self._audio_session is not None
        has_device = self.device_session is not None
        has_meter = self.device_session.meter if has_device else None
        volume = self.get_volume_percent() if has_meter else None
        return (
            f"VolumeMonitor("
            f"process_name={self.process_name}, "
            f"has_session={has_session}, "
            f"device_session={has_device}, "
            f"has_meter={has_meter is not None}, "
            f"volume_gain={self.volume_gain}, "
            f"volume={volume}%)"
        )


class VolumeRecorder:
    """
    音量记录器类。

    负责记录音量数据，维护时间窗口内的音量历史，并提供统计功能。
    可以持有 VolumeMonitor 实例并自动记录其监控到的音量数据。
    """

    def __init__(
        self,
        monitor: Optional["VolumeMonitor"] = None,
        window_duration_ms: int = 5000,
    ):
        """
        初始化音量记录器。

        :param monitor: VolumeMonitor 实例，如果为 None 则创建默认实例
        :param window_duration_ms: 时间窗口长度（毫秒），默认 5000ms
        """
        self.window_duration_ms = window_duration_ms
        # 使用 deque 存储 (timestamp, volume) 元组
        self._volumes: deque[tuple[float, int]] = deque()
        # 历史统计（全局）
        self._history_min: Optional[int] = None
        self._history_max: Optional[int] = None
        self._history_sum: int = 0
        self._history_count: int = 0
        # 记录状态标志
        self._is_recording = False

        # 设置 monitor 实例
        if monitor is None:
            # 创建默认的 monitor 实例（会自动创建 device_session）
            self.monitor = VolumeMonitor(volume_gain=VOLUME_GAIN)
        else:
            self.monitor = monitor

    def record(self, volume: int, timestamp: Optional[float] = None):
        """
        记录一个音量数据点。

        :param volume: 音量百分比（0-100）
        :param timestamp: 时间戳（秒），如果为 None 则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()

        # 添加新数据点
        self._volumes.append((timestamp, volume))

        # 更新历史统计
        if self._history_min is None or volume < self._history_min:
            self._history_min = volume
        if self._history_max is None or volume > self._history_max:
            self._history_max = volume
        self._history_sum += volume
        self._history_count += 1

        # 清理过期数据
        self._cleanup_old_data(timestamp)

    def _cleanup_old_data(self, current_time: float):
        """
        清理超出时间窗口的旧数据。

        :param current_time: 当前时间戳（秒）
        """
        window_start = current_time - self.window_duration_ms / 1000
        while self._volumes and self._volumes[0][0] < window_start:
            self._volumes.popleft()

    def get_window_volumes(self) -> list[int]:
        """
        获取当前时间窗口内的所有音量数据。

        :return: 音量列表
        """
        return [vol for _, vol in self._volumes]

    def get_window_stats(self) -> tuple[Optional[int], Optional[float], Optional[int]]:
        """
        获取当前时间窗口内的音量统计。

        :return: (min, avg, max)，如果窗口为空则返回 (None, None, None)
        """
        volumes = self.get_window_volumes()
        if not volumes:
            return None, None, None
        return min(volumes), sum(volumes) / len(volumes), max(volumes)

    def get_history_stats(self) -> tuple[Optional[int], Optional[float], Optional[int]]:
        """
        获取历史音量统计（所有记录过的数据）。

        :return: (min, avg, max)，如果没有历史数据则返回 (None, None, None)
        """
        if self._history_count == 0:
            return None, None, None
        history_avg = self._history_sum / self._history_count
        return self._history_min, history_avg, self._history_max

    def get_window_min(self) -> Optional[int]:
        """
        获取当前时间窗口内的最小音量。

        :return: 最小音量，窗口为空则返回 None
        """
        volumes = self.get_window_volumes()
        return min(volumes) if volumes else None

    def get_window_size(self) -> int:
        """
        获取当前时间窗口内的数据点数量。

        :return: 数据点数量
        """
        return len(self._volumes)

    def clear(self):
        """清空所有记录数据。"""
        self._volumes.clear()
        self._history_min = None
        self._history_max = None
        self._history_sum = 0
        self._history_count = 0

    def start(self, log_output: bool = True):
        """
        开始监控音量并记录数据。

        :param log_output: 是否输出日志（包含音量字符和统计信息）
        """
        if not self.monitor.device_session or not self.monitor.device_session.meter:
            logger.fail("无法开始监控: 音频设备会话未设置")
            return

        if self._is_recording:
            logger.warn("记录器已在运行中")
            return

        self._is_recording = True

        if log_output:
            self.monitor._log_monitor_start()

        sample_count = 0
        group_volumes = []

        try:
            while self._is_recording:
                volume_percent = self.monitor.get_volume_percent()
                current_time = time.time()

                if volume_percent is None:
                    if log_output:
                        logger.warn("无法获取音量，跳过此次采样")
                    self.monitor.sleep_interval()
                    continue

                # 记录音量数据
                self.record(volume_percent, current_time)

                if log_output:
                    group_volumes.append(volume_percent)
                    volume_char = self.monitor.get_volume_char(volume_percent)
                    self.monitor._log_volume_char(volume_char, sample_count)
                    if self.monitor._is_last_in_group(sample_count):
                        self.monitor._log_group_stats(group_volumes)
                        group_volumes = []

                sample_count += 1
                self.monitor.sleep_interval()

        except KeyboardInterrupt:
            if log_output and sample_count % self.monitor.samples_per_group != 0:
                self.monitor._log_group_stats(group_volumes)
            logger.note("记录已停止")
        except Exception as e:
            logger.warn(f"记录过程中出错: {e}")
        finally:
            self._is_recording = False

    def stop(self):
        """
        停止记录。
        """
        if not self._is_recording:
            logger.warn("记录器未在运行")
            return
        self._is_recording = False
        logger.okay("记录器已停止")

    @property
    def is_recording(self) -> bool:
        """获取记录器运行状态。"""
        return self._is_recording

    def __repr__(self) -> str:
        window_stats = self.get_window_stats()
        history_stats = self.get_history_stats()
        has_monitor = self.monitor is not None
        return (
            f"VolumeRecorder("
            f"window_duration={self.window_duration_ms}ms, "
            f"window_size={self.get_window_size()}, "
            f"window_stats={window_stats}, "
            f"history_stats={history_stats}, "
            f"has_monitor={has_monitor})"
        )


def test_volume_monitor():
    """测试音量监控器"""
    # 创建音频设备会话
    device_session = AudioDeviceSession(device_name=GTAV_AUDIO_DEVICE_NAME)
    # 设置音频设备
    if not device_session.setup():
        logger.fail("无法设置音频设备，监控终止")
        return
    logger.note(f"设备会话信息: {device_session}")
    # 创建音量监控器
    monitor = VolumeMonitor(device_session=device_session, volume_gain=VOLUME_GAIN)
    # 可选：检查游戏是否运行
    monitor.find_audio_session()
    logger.note(f"音量监控器信息: {monitor}")
    volume = monitor.get_volume_percent()
    if volume is not None:
        logger.note(f"当前音量: {volume}%")
        logger.note(f"音量字符: {monitor.get_volume_char(volume)}")
    # 开始持续监控
    monitor.start()


def test_volume_recorder():
    """测试音量记录器"""
    # 创建音量记录器（自动创建默认 monitor）
    recorder = VolumeRecorder(window_duration_ms=5000)
    logger.note(f"音量记录器信息: {recorder}")

    # 可选：检查游戏是否运行
    recorder.monitor.find_audio_session()
    logger.note(f"音量监控器信息: {recorder.monitor}")

    # 监控（持续运行，按 Ctrl+C 停止）
    logger.note("开始监控...")
    recorder.start()

    # 输出统计信息
    logger.note("=" * 50)
    logger.note("音量统计信息")
    logger.note("=" * 50)
    window_stats = recorder.get_window_stats()
    history_stats = recorder.get_history_stats()
    window_avg = f"{window_stats[1]:.1f}" if window_stats[1] is not None else "None"
    history_avg = f"{history_stats[1]:.1f}" if history_stats[1] is not None else "None"
    logger.note(
        f"窗口统计 (最近 5 秒): min={window_stats[0]}, avg={window_avg}, max={window_stats[2]}"
    )
    logger.note(
        f"历史统计 (全部数据): min={history_stats[0]}, avg={history_avg}, max={history_stats[2]}"
    )
    logger.note(f"窗口数据点数量: {recorder.get_window_size()}")
    logger.note(f"记录器信息: {recorder}")


if __name__ == "__main__":
    # 测试音量监控器
    # test_volume_monitor()

    # 测试音量记录器
    test_volume_recorder()

    # python -m gtaz.audios.volumes
