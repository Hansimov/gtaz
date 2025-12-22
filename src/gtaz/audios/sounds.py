"""音频实时采样和录制模块

安装依赖：

```sh
pip install sounddevice soundfile
```
"""

import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from tclogger import TCLogger, logstr


logger = TCLogger(name="SoundRecorder", use_prefix=True, use_prefix_ms=True)


# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parents[1]
# 缓存目录
CACHE_DIR = MODULE_DIR / "cache"
# 录音目录
SOUNDS_DIR = CACHE_DIR / "sounds"

# 音频设备名称（CABLE Output 是虚拟线缆的输出端，用于录制其他应用的音频）
AUDIO_DEVICE_NAME = "CABLE Output"
# 默认采样率
SAMPLE_RATE = 44100
# 默认通道数
CHANNELS = 2
# 默认音频格式
AUDIO_FORMAT = "wav"
# 窗口时长（毫秒）
WINDOW_MS = 6000
# 默认录制时长（秒）
DEFAULT_DURATION = 10
# 最大录制时长（秒）
MAX_DURATION = 600

# 录制启停热键
START_RECORD_KEY = "1"
STOP_RECORD_KEY = "2"

# 进度日志样式映射
PROGRESS_LOGSTR = {
    0: logstr.file,
    25: logstr.mesg,
    50: logstr.note,
    75: logstr.hint,
    100: logstr.okay,
}


def get_progress_logstr(percent: float):
    """根据百分比获取对应的日志样式函数。"""
    for threshold in sorted(PROGRESS_LOGSTR.keys(), reverse=True):
        if percent >= threshold:
            return PROGRESS_LOGSTR[threshold]
    return logstr.file


def brq(s) -> str:
    """为字符串添加单引号。"""
    return f"'{s}'"


def key_hint(s) -> str:
    """为按键添加提示样式。"""
    return logstr.hint(brq(s))


def val_mesg(s) -> str:
    """为值添加消息样式。"""
    return logstr.mesg(s)


def list_audio_devices():
    """
    列出所有音频设备。
    """
    try:
        devices = sd.query_devices()
        logger.note("可用音频设备列表:")
        for i, device in enumerate(devices):
            in_ch = device["max_input_channels"]
            out_ch = device["max_output_channels"]
            ch_info = f"in={in_ch}, out={out_ch}"
            logger.mesg(f"  [{i}] {device['name']} ({ch_info})")
    except Exception as e:
        logger.warn(f"列出音频设备时出错: {e}")


def find_audio_device(device_name: str = AUDIO_DEVICE_NAME) -> Optional[int]:
    """
    查找音频设备索引。

    :param device_name: 设备名称关键字

    :return: 设备索引，未找到则返回 None
    """
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in device["name"].lower():
                if device["max_input_channels"] > 0:
                    logger.okay(
                        f"找到音频设备: [{i}] {device['name']} "
                        f"(输入通道: {device['max_input_channels']})"
                    )
                    return i
        logger.warn(f"未找到包含 '{device_name}' 的输入设备")
        list_audio_devices()
        return None
    except Exception as e:
        logger.warn(f"查找音频设备时出错: {e}")
        return None


@dataclass
class AudioChunk:
    """
    音频数据块。

    用于存储单个音频数据块及其元信息。
    """

    data: np.ndarray
    """音频数据 (samples, channels)"""
    timestamp: float
    """采集时间戳"""
    sample_rate: int
    """采样率"""
    channels: int
    """通道数"""

    @property
    def duration_ms(self) -> float:
        """获取数据块时长（毫秒）。"""
        return len(self.data) / self.sample_rate * 1000

    @property
    def samples(self) -> int:
        """获取采样点数。"""
        return len(self.data)


class AudioBuffer:
    """
    环形音频缓冲区。

    使用 deque 实现固定时长的音频数据缓冲区，支持实时采集和读取。
    """

    def __init__(
        self,
        window_ms: int = WINDOW_MS,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
    ):
        """
        初始化音频缓冲区。

        :param window_ms: 窗口时长（毫秒）
        :param sample_rate: 采样率
        :param channels: 通道数
        """
        self.window_ms = window_ms
        self.sample_rate = sample_rate
        self.channels = channels

        # 计算窗口内最大采样点数
        self._max_samples = int(window_ms / 1000 * sample_rate)
        # 使用 deque 存储音频块
        self._chunks: deque[AudioChunk] = deque()
        # 当前缓冲区内的总采样点数
        self._total_samples: int = 0
        # 线程锁
        self._lock = threading.Lock()

    def add_chunk(self, data: np.ndarray, timestamp: float = None):
        """
        添加音频数据块到缓冲区。

        :param data: 音频数据 (samples, channels)
        :param timestamp: 时间戳，默认为当前时间
        """
        if timestamp is None:
            timestamp = time.time()

        chunk = AudioChunk(
            data=data.copy(),
            timestamp=timestamp,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

        with self._lock:
            self._chunks.append(chunk)
            self._total_samples += chunk.samples
            # 清理超出窗口的旧数据
            self._cleanup()

    def _cleanup(self):
        """清理超出时间窗口的旧数据。"""
        while self._total_samples > self._max_samples and self._chunks:
            old_chunk = self._chunks.popleft()
            self._total_samples -= old_chunk.samples

    def get_window_data(self) -> Optional[np.ndarray]:
        """
        获取当前窗口内的所有音频数据。

        :return: 音频数据 (samples, channels)，缓冲区为空则返回 None
        """
        with self._lock:
            if not self._chunks:
                return None
            # 合并所有数据块
            all_data = [chunk.data for chunk in self._chunks]
            return np.concatenate(all_data, axis=0)

    def get_window_duration_ms(self) -> float:
        """
        获取当前窗口内的实际时长（毫秒）。

        :return: 窗口时长
        """
        with self._lock:
            return self._total_samples / self.sample_rate * 1000

    def get_sample_count(self) -> int:
        """
        获取当前缓冲区内的采样点数。

        :return: 采样点数
        """
        with self._lock:
            return self._total_samples

    def clear(self):
        """清空缓冲区。"""
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0

    def __len__(self) -> int:
        return self.get_sample_count()

    def __repr__(self) -> str:
        duration = self.get_window_duration_ms()
        return (
            f"AudioBuffer("
            f"window_ms={self.window_ms}, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"samples={len(self)}, "
            f"duration={duration:.1f}ms)"
        )


class SoundRecorder:
    """
    音频录制器。

    实时录制指定音频设备的声音，支持缓冲区管理和文件保存。
    """

    def __init__(
        self,
        device_name: str = AUDIO_DEVICE_NAME,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        window_ms: int = WINDOW_MS,
        output_dir: Path = None,
        audio_format: str = AUDIO_FORMAT,
    ):
        """
        初始化音频录制器。

        :param device_name: 音频设备名称
        :param sample_rate: 采样率
        :param channels: 通道数
        :param window_ms: 缓冲区窗口时长（毫秒）
        :param output_dir: 输出目录
        :param audio_format: 音频格式（wav/flac/ogg）
        """
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_ms = window_ms
        self.audio_format = audio_format

        # 查找音频设备
        self.device_index = find_audio_device(device_name)
        if self.device_index is None:
            logger.warn(f"未找到音频设备 '{device_name}'，将使用默认设备")

        # 设置输出目录
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_dir is None:
            output_dir = SOUNDS_DIR
        self.save_dir = Path(output_dir) / session_name

        # 初始化缓冲区
        self.buffer = AudioBuffer(
            window_ms=window_ms,
            sample_rate=sample_rate,
            channels=channels,
        )

        # 录制状态
        self._is_recording = False
        self._recorded_chunks: list[AudioChunk] = []
        self._record_start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

        # 数据回调（可用于实时处理）
        self._data_callback: Optional[Callable[[np.ndarray, float], Any]] = None

    def set_data_callback(self, callback: Callable[[np.ndarray, float], Any]):
        """
        设置数据回调函数。

        :param callback: 回调函数，接收 (data, timestamp) 参数
        """
        self._data_callback = callback

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        音频流回调函数。

        :param indata: 输入音频数据
        :param frames: 帧数
        :param time_info: 时间信息
        :param status: 状态信息
        """
        if status:
            logger.warn(f"音频流状态: {status}")

        timestamp = time.time()
        data = indata.copy()

        # 添加到缓冲区
        self.buffer.add_chunk(data, timestamp)

        # 如果正在录制，保存数据块
        if self._is_recording:
            with self._lock:
                chunk = AudioChunk(
                    data=data,
                    timestamp=timestamp,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )
                self._recorded_chunks.append(chunk)

        # 调用数据回调
        if self._data_callback:
            try:
                self._data_callback(data, timestamp)
            except Exception as e:
                logger.warn(f"数据回调执行出错: {e}")

    def start_stream(self) -> bool:
        """
        启动音频流（开始采集音频到缓冲区）。

        :return: 是否成功启动
        """
        if self._stream is not None:
            logger.warn("音频流已在运行")
            return True

        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                dtype=np.float32,
            )
            self._stream.start()
            logger.okay(f"音频流已启动 (设备: {self.device_name})")
            return True
        except Exception as e:
            logger.warn(f"启动音频流失败: {e}")
            return False

    def stop_stream(self):
        """停止音频流。"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.okay("音频流已停止")

    def start_recording(self):
        """开始录制（在音频流运行的基础上开始保存数据）。"""
        if self._is_recording:
            logger.warn("已在录制中")
            return

        with self._lock:
            self._recorded_chunks.clear()
            self._record_start_time = time.time()
            self._is_recording = True

        logger.okay("开始录制...")

    def stop_recording(self) -> float:
        """
        停止录制。

        :return: 录制时长（秒）
        """
        if not self._is_recording:
            logger.warn("未在录制中")
            return 0.0

        with self._lock:
            self._is_recording = False
            duration = (
                time.time() - self._record_start_time
                if self._record_start_time
                else 0.0
            )

        logger.okay(f"停止录制，时长: {duration:.2f} 秒")
        return duration

    def _generate_filename(self) -> str:
        """
        生成录音文件名。

        :return: 文件名字符串
        """
        now = datetime.now()
        return (
            now.strftime("%Y-%m-%d_%H-%M-%S-")
            + f"{now.microsecond // 1000:03d}.{self.audio_format}"
        )

    def get_recorded_data(self) -> Optional[np.ndarray]:
        """
        获取已录制的音频数据。

        :return: 音频数据 (samples, channels)，无数据则返回 None
        """
        with self._lock:
            if not self._recorded_chunks:
                return None
            all_data = [chunk.data for chunk in self._recorded_chunks]
            return np.concatenate(all_data, axis=0)

    def get_recorded_duration(self) -> float:
        """
        获取已录制的时长（秒）。

        :return: 录制时长
        """
        with self._lock:
            total_samples = sum(chunk.samples for chunk in self._recorded_chunks)
            return total_samples / self.sample_rate

    def save_recording(
        self, filename: str = None, verbose: bool = True
    ) -> Optional[Path]:
        """
        保存录制的音频到文件。

        :param filename: 文件名，默认自动生成
        :param verbose: 是否打印日志

        :return: 保存的文件路径，失败则返回 None
        """
        data = self.get_recorded_data()
        if data is None or len(data) == 0:
            logger.warn("没有录制数据可保存")
            return None

        # 创建输出目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if filename is None:
            filename = self._generate_filename()
        file_path = self.save_dir / filename

        # 输出音频统计信息（用于调试）
        if verbose:
            max_val = np.max(np.abs(data))
            mean_val = np.mean(np.abs(data))
            logger.note(
                f"音频统计: max={max_val:.6f}, mean={mean_val:.6f}, shape={data.shape}"
            )

        # 保存音频文件
        try:
            sf.write(file_path, data, self.sample_rate)
            if verbose:
                duration = len(data) / self.sample_rate
                logger.okay(f"音频已保存: {file_path} ({duration:.2f}s)")
            return file_path
        except Exception as e:
            logger.warn(f"保存音频失败: {e}")
            return None

    def save_window(self, filename: str = None, verbose: bool = True) -> Optional[Path]:
        """
        保存当前缓冲区窗口内的音频。

        :param filename: 文件名，默认自动生成
        :param verbose: 是否打印日志

        :return: 保存的文件路径，失败则返回 None
        """
        data = self.buffer.get_window_data()
        if data is None or len(data) == 0:
            logger.warn("缓冲区为空")
            return None

        # 创建输出目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if filename is None:
            filename = self._generate_filename()
        file_path = self.save_dir / filename

        # 保存音频文件
        try:
            sf.write(file_path, data, self.sample_rate)
            if verbose:
                duration = len(data) / self.sample_rate
                logger.okay(f"窗口音频已保存: {file_path} ({duration:.2f}s)")
            return file_path
        except Exception as e:
            logger.warn(f"保存窗口音频失败: {e}")
            return None

    def clear_recording(self):
        """清空录制数据。"""
        with self._lock:
            self._recorded_chunks.clear()
            self._record_start_time = None

    @property
    def is_recording(self) -> bool:
        """获取录制状态。"""
        return self._is_recording

    @property
    def is_streaming(self) -> bool:
        """获取音频流状态。"""
        return self._stream is not None and self._stream.active

    def __del__(self):
        """析构函数，确保停止音频流。"""
        self.stop_stream()

    def __repr__(self) -> str:
        return (
            f"SoundRecorder("
            f"device={self.device_name}, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"window_ms={self.window_ms}, "
            f"format={self.audio_format}, "
            f"is_recording={self.is_recording}, "
            f"is_streaming={self.is_streaming}, "
            f"recorded_duration={self.get_recorded_duration():.2f}s)"
        )


class RecordRunner:
    """
    录制运行器。

    控制录制的启动、停止和保存，支持定时录制和热键控制。
    """

    def __init__(
        self,
        recorder: SoundRecorder,
        duration: float = None,
        hotkey_toggle: bool = False,
        exit_after_record: bool = False,
    ):
        """
        初始化录制运行器。

        :param recorder: 音频录制器实例
        :param duration: 录制时长（秒），None 表示持续录制
        :param hotkey_toggle: 是否启用热键启停模式
        :param exit_after_record: 录制后是否退出
        """
        self.recorder = recorder
        self.duration = duration
        self.hotkey_toggle = hotkey_toggle
        self.exit_after_record = exit_after_record

        # 热键检测器（可选，需要 keyboards 模块）
        self.start_detector = None
        self.stop_detector = None
        if hotkey_toggle:
            self._create_detectors()

    def _create_detectors(self):
        """创建热键检测器。"""
        try:
            from ..devices.keyboards import KeyboardActionDetector, KEY_DOWN

            self.start_detector = KeyboardActionDetector(
                monitor_keys=[START_RECORD_KEY], trigger_type=KEY_DOWN
            )
            self.stop_detector = KeyboardActionDetector(
                monitor_keys=[STOP_RECORD_KEY], trigger_type=KEY_DOWN
            )
        except ImportError:
            logger.warn("无法导入 keyboards 模块，热键功能不可用")
            self.hotkey_toggle = False

    def _log_keyboard_interrupt(self):
        logger.note(f"\n检测到 {key_hint('Ctrl+C')}，正在退出...")

    def _wait_start_signal(self) -> bool:
        """
        等待热键启动信号。

        :return: 是否收到启动信号
        """
        if not self.start_detector:
            return True

        logger.note("热键启停模式已启动")
        logger.note(
            f"按 {key_hint(START_RECORD_KEY)} {val_mesg('开始录制')}，"
            f"按 {key_hint(STOP_RECORD_KEY)} {val_mesg('停止录制')}，"
            f"按 {key_hint('Ctrl+C')} {val_mesg('退出')}"
        )

        while True:
            action_info = self.start_detector.detect()
            if action_info.has_action:
                logger.okay(f"检测到 {key_hint(START_RECORD_KEY)} 键，开始录制...")
                return True
            time.sleep(0.015)

    def _log_loop_progress(self, elapsed: float):
        """循环进度日志。"""
        if self.duration is None or self.duration <= 0:
            logger.okay(f"录制中: {elapsed:.1f}s")
            return

        percent = min(100, (elapsed / self.duration) * 100)
        progress_logstr = get_progress_logstr(percent)
        progress_str = progress_logstr(
            f"({percent:5.1f}%) [{elapsed:.1f}/{self.duration:.1f}]"
        )
        logger.okay(f"{progress_str} 录制中...")

    def _log_duration(self):
        """日志输出录制时长信息。"""
        if self.duration is None or self.duration <= 0:
            self.duration = MAX_DURATION
            if self.stop_detector:
                logger.note(
                    f"持续模式：按 {key_hint(STOP_RECORD_KEY)} {val_mesg('停止录制')}..."
                )
            else:
                logger.note(
                    f"持续模式：按 {key_hint('Ctrl+C')} {val_mesg('停止录制')}..."
                )
        else:
            logger.note(f"定时模式：{self.duration} 秒...")

    def _save_recording(self):
        """保存录制数据。"""
        duration = self.recorder.get_recorded_duration()
        if duration > 0:
            logger.note(f"录制完成，时长 {duration:.2f} 秒，开始保存...")
            file_path = self.recorder.save_recording(verbose=True)
            if file_path:
                logger.okay(f"保存完成: {file_path}")

    def _run_loop(self):
        """运行录制循环。"""
        # 热键启停模式：等待启动信号
        if self.start_detector:
            self._wait_start_signal()

        # 日志输出录制时长
        self._log_duration()

        # 开始录制
        self.recorder.start_recording()
        start_time = time.time()

        # 主循环
        interrupted = False
        try:
            while True:
                elapsed = time.time() - start_time

                # 检查是否达到录制时长
                if self.duration and elapsed >= self.duration:
                    break

                # 检查停止键
                if self.stop_detector:
                    action_info = self.stop_detector.detect()
                    if action_info.has_action:
                        logger.note(
                            f"检测到 {key_hint(STOP_RECORD_KEY)} 键，停止录制..."
                        )
                        break

                # 输出进度
                if int(elapsed) % 1 == 0:  # 每秒输出一次
                    self._log_loop_progress(elapsed)

                time.sleep(0.1)

        except KeyboardInterrupt:
            interrupted = True
            self._log_keyboard_interrupt()

        # 停止录制
        self.recorder.stop_recording()

        # 保存录制数据
        self._save_recording()

        # 清空录制数据
        self.recorder.clear_recording()

        # 如果是被中断的，重新抛出异常让 run() 捕获
        if interrupted:
            raise KeyboardInterrupt

    def run(self):
        """运行录制器。"""
        # 启动音频流
        if not self.recorder.start_stream():
            logger.warn("无法启动音频流，录制终止")
            return

        logger.note(f"录制器信息: {self.recorder}")

        try:
            if self.exit_after_record:
                self._run_loop()
            else:
                while True:
                    self._run_loop()
        except KeyboardInterrupt:
            self._log_keyboard_interrupt()
        finally:
            # 停止音频流
            self.recorder.stop_stream()


class SoundRecorderArgParser:
    """音频录制器命令行参数解析器。"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GTAV 音频录制器")
        self._add_arguments()

    def _add_arguments(self):
        """添加命令行参数。"""
        self.parser.add_argument(
            "-l",
            "--list-devices",
            action="store_true",
            help="列出所有可用的音频设备",
        )
        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=None,
            help="录制时长（秒），不指定则持续录制",
        )
        self.parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default=None,
            help="音频文件保存目录",
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
            "-c",
            "--channels",
            type=int,
            default=CHANNELS,
            help=f"通道数（默认: {CHANNELS}）",
        )
        self.parser.add_argument(
            "-f",
            "--format",
            type=str,
            default=AUDIO_FORMAT,
            choices=["wav", "flac", "ogg"],
            help=f"音频格式（默认: {AUDIO_FORMAT}）",
        )
        self.parser.add_argument(
            "-w",
            "--window-ms",
            type=int,
            default=WINDOW_MS,
            help=f"缓冲区窗口时长（毫秒，默认: {WINDOW_MS}）",
        )
        self.parser.add_argument(
            "-g",
            "--hotkey-toggle",
            action="store_true",
            help=f"使用热键启停录制，按 '{START_RECORD_KEY}' 开始，按 '{STOP_RECORD_KEY}' 停止",
        )
        self.parser.add_argument(
            "-x",
            "--exit-after-record",
            action="store_true",
            default=False,
            help="录制后退出（默认不退出，继续监听新的录制触发事件）",
        )

    def parse(self) -> argparse.Namespace:
        """解析命令行参数。"""
        return self.parser.parse_args()


def main():
    """命令行入口。"""
    args = SoundRecorderArgParser().parse()

    # 列出设备模式
    if args.list_devices:
        list_audio_devices()
        return

    # 创建录制器
    recorder = SoundRecorder(
        device_name=args.device_name,
        sample_rate=args.sample_rate,
        channels=args.channels,
        window_ms=args.window_ms,
        output_dir=args.output_dir,
        audio_format=args.format,
    )

    # 当指定了 duration 参数时，默认录制后退出（除非明确指定了 -x 参数）
    # 如果没有指定 duration，则使用 exit_after_record 参数的值
    exit_after_record = args.exit_after_record
    if args.duration is not None and not args.hotkey_toggle:
        # 定时录制模式默认录制完成后退出
        exit_after_record = True

    # 创建录制运行器
    runner = RecordRunner(
        recorder=recorder,
        duration=args.duration,
        hotkey_toggle=args.hotkey_toggle,
        exit_after_record=exit_after_record,
    )

    # 运行
    runner.run()


if __name__ == "__main__":
    main()

    # Case: 列出所有音频设备
    # python -m gtaz.audios.sounds -l

    # Case: 录制 10 秒
    # python -m gtaz.audios.sounds -d 10

    # Case: 持续录制（Ctrl+C 停止）
    # python -m gtaz.audios.sounds

    # Case: 热键启停模式
    # python -m gtaz.audios.sounds -g

    # Case: 指定设备和格式
    # python -m gtaz.audios.sounds -n "CABLE Output" -f flac -d 30

    # Case: 自定义采样率和通道数
    # python -m gtaz.audios.sounds -r 48000 -c 1 -d 10
