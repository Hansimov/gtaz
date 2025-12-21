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
# 音量字符（6个高度）
VOLUME_CHARS = "▁▂▃▅▆▇"
VOLUME_BITS = len(VOLUME_CHARS)
# 采样间隔（毫秒）
SAMPLE_INTERVAL_MS = 200
# 每组输出的样本数
SAMPLES_PER_GROUP = 50
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

    使用 AudioDeviceSession 进行音量监控，支持音量增益。
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
        self.device_session = device_session
        self.process_name = process_name
        self.sample_interval_ms = sample_interval_ms
        self.samples_per_group = samples_per_group
        self.volume_gain = volume_gain
        self._audio_session = None

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

    def monitor_volume(self, duration_seconds: Optional[float] = None):
        """
        监控音量并输出到日志。
        :param duration_seconds: 监控持续时间（秒），None 表示持续监控
        """
        if not self.device_session or not self.device_session.meter:
            logger.fail("无法开始监控: 音频设备会话未设置")
            return
        self._log_monitor_start()
        start_time = time.time()
        sample_count = 0
        group_volumes = []
        try:
            while True:
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
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


def test_gtav_volume_monitor():
    """测试 GTAV 音量监控器。"""
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
    monitor.monitor_volume()


if __name__ == "__main__":
    test_gtav_volume_monitor()

    # python -m gtaz.audios.volumes
