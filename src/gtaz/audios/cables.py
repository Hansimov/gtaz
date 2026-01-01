"""GTAV 音频设备切换模块

使用 svcl.exe（SoundVolumeCommandLine）设置 GTAV 的音频输入/输出设备。

功能:
- 将 GTAV 的音频输出设置为 CABLE Input（虚拟音频线缆）
- 将 GTAV 的音频输入设置为 CABLE Output（虚拟音频线缆）
- 或恢复为系统默认设备
- 查看当前的音频输入输出设备

svcl.exe 命令行参考:
- /SetAppDefault [Device Name] [Default Type] [Process Name/ID]
  Default Type: 0=Console, 1=Multimedia, 2=Communications, all=All
- /scomma "" - 导出 CSV 格式的音频设备列表

"""

import argparse
import csv
import io
import subprocess
import sys

from pathlib import Path
from tclogger import TCLogger, logstr, brk


logger = TCLogger(name="AudioDeviceSwitcher", use_prefix=True, use_prefix_ms=True)


# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parent
# svcl.exe 路径
SVCL_EXE = MODULE_DIR / "svcl.exe"

# GTAV 增强版应用名称（在 SoundVolumeView 中显示的名称）
GTAV_APP_NAME = "Grand Theft Auto V"
# GTAV 增强版进程名（可执行文件名）
GTAV_PROCESS_NAME = "GTA5_Enhanced.exe"

# CABLE 虚拟音频设备名称
CABLE_INPUT_DEVICE = "CABLE Input"  # 音频输出设备（输出到虚拟线缆的输入端）
CABLE_OUTPUT_DEVICE = "CABLE Output"  # 音频输入设备（从虚拟线缆的输出端输入）

# 系统默认设备标识（SoundVolumeView 使用空字符串表示默认设备）
DEFAULT_RENDER_DEVICE = "DefaultRenderDevice"
DEFAULT_CAPTURE_DEVICE = "DefaultCaptureDevice"


def run_svcl_command(
    args_list: list[str], show_cmd: bool = False
) -> tuple[bool, str, str]:
    """
    执行 svcl.exe 命令。

    :param args_list: 命令参数列表
    :param show_cmd: 是否在日志中显示命令

    :return: (是否成功, stdout, stderr)
    """
    if not SVCL_EXE.exists():
        logger.err(f"未找到 svcl.exe: {SVCL_EXE}")
        return False, "", "svcl.exe not found"

    cmd_list = [str(SVCL_EXE)] + args_list
    cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd_list)

    try:
        if show_cmd:
            logger.mesg("执行命令:")
            logger.file(cmd_str)

        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            encoding="utf-8-sig",  # 处理 BOM
            errors="ignore",
            shell=False,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        # svcl.exe 通常不输出内容，成功时返回码为 0
        return result.returncode == 0, stdout, stderr

    except Exception as e:
        logger.err(f"执行命令时出错: {e}")
        return False, "", str(e)


class AudioDeviceSwitcher:
    """
    GTAV 音频设备切换器。

    使用 svcl.exe（SoundVolumeCommandLine）管理 GTAV 的音频输入/输出设备设置。
    """

    def __init__(
        self, app_name: str = GTAV_APP_NAME, process_name: str = GTAV_PROCESS_NAME
    ):
        """
        初始化音频设备切换器。

        :param app_name: 应用名称（用于显示和查找）
        :param process_name: 进程名（用于设置音频设备）
        """
        self.app_name = app_name
        self.process_name = process_name

    @property
    def app_str(self) -> str:
        """获取应用名称的日志字符串表示。"""
        return logstr.file(brk(self.app_name))

    def _log_title(self, op: str):
        """打印操作标题。"""
        logger.note("=" * 50)
        logger.note(f"{logstr.note(op)}音频设备")
        logger.note("=" * 50)

    def _log_device_setting(self, direction: str, device: str):
        """打印设备设置日志。"""
        direction_str = logstr.mesg(direction)
        device_str = logstr.file(device)
        logger.mesg(f"{direction_str}: {device_str}")

    def set_output_device(self, device_name: str) -> bool:
        """
        设置音频输出设备（Render）。

        :param device_name: 设备名称
        :return: 是否设置成功
        """
        # /SetAppDefault [Device Name] [Default Type] [Process Name/ID]
        # Default Type: 0=Console, 1=Multimedia, 2=Communications, all=All
        # 使用进程名（如 GTA5_Enhanced.exe）而不是窗口标题
        args = ["/SetAppDefault", device_name, "all", self.process_name]
        success, stdout, stderr = run_svcl_command(args)

        if success:
            self._log_device_setting("音频输出", device_name)
        else:
            logger.warn(f"设置音频输出设备失败")
            if stderr:
                logger.warn(stderr)

        return success

    def set_input_device(self, device_name: str) -> bool:
        """
        设置音频输入设备（Capture）。

        :param device_name: 设备名称
        :return: 是否设置成功
        """
        # /SetAppDefault [Device Name] [Default Type] [Process Name/ID]
        # Default Type: 0=Console, 1=Multimedia, 2=Communications, all=All
        # 注意：GTAV 通常只有输出（Render）设备，可能没有输入（Capture）设备
        # 使用进程名（如 GTA5_Enhanced.exe）而不是窗口标题
        args = ["/SetAppDefault", device_name, "all", self.process_name]
        success, stdout, stderr = run_svcl_command(args)

        if success:
            self._log_device_setting("音频输入", device_name)
        else:
            logger.warn(f"设置音频输入设备失败")
            if stderr:
                logger.warn(stderr)

        return success

    def set_cable(self) -> bool:
        """
        设置为 CABLE 虚拟音频设备。

        将 GTAV 的音频输出设置为 CABLE Input，
        音频输入设置为 CABLE Output。

        :return: 是否全部设置成功
        """
        self._log_title("设置 CABLE")
        logger.mesg(f"应用: {self.app_str}")

        output_success = self.set_output_device(CABLE_INPUT_DEVICE)
        input_success = self.set_input_device(CABLE_OUTPUT_DEVICE)

        if output_success and input_success:
            logger.okay("CABLE 音频设备设置完成")
        else:
            logger.warn("部分设备设置失败")

        return output_success and input_success

    def set_default(self) -> bool:
        """
        设置为系统默认音频设备。

        :return: 是否全部设置成功
        """
        self._log_title("设置默认")
        logger.mesg(f"应用: {self.app_str}")

        output_success = self.set_output_device(DEFAULT_RENDER_DEVICE)
        input_success = self.set_input_device(DEFAULT_CAPTURE_DEVICE)

        if output_success and input_success:
            logger.okay("默认音频设备设置完成")
        else:
            logger.warn("部分设备设置失败")

        return output_success and input_success

    def get_current_devices(self) -> dict[str, str]:
        """
        获取应用当前的音频输入输出设备。

        :return: 包含 'output' 和 'input' 键的字典，值为设备名称
        """
        # 执行 svcl.exe 导出 CSV
        args = ["/scomma", ""]
        success, stdout, stderr = run_svcl_command(args)

        if not success:
            logger.warn("获取音频设备信息失败")
            return {"output": None, "input": None}

        # 解析 CSV 输出
        result = {"output": None, "input": None}
        try:
            csv_reader = csv.DictReader(io.StringIO(stdout))
            for row in csv_reader:
                # 筛选条件：
                # 1. Name 包含应用名称
                # 2. Type 为 Application
                # 3. Device State 为 Active
                if (
                    self.app_name in row.get("Name", "")
                    and row.get("Type") == "Application"
                    and row.get("Device State") == "Active"
                ):
                    direction = row.get("Direction")
                    device_name = row.get("Device Name")

                    if direction == "Render":
                        result["output"] = device_name
                    elif direction == "Capture":
                        result["input"] = device_name

        except Exception as e:
            logger.err(f"解析音频设备信息时出错: {e}")

        return result

    def list_current_devices(self) -> bool:
        """
        列出应用当前的音频输入输出设备。

        :return: 是否成功获取设备信息
        """
        self._log_title("查看当前")
        logger.mesg(f"应用: {self.app_str}")

        devices = self.get_current_devices()

        if devices["output"] is not None:
            self._log_device_setting("音频输出", devices["output"])
        else:
            logger.warn("未找到活动的音频输出设备")

        if devices["input"] is not None:
            self._log_device_setting("音频输入", devices["input"])
        else:
            logger.warn("未找到活动的音频输入设备")

        return devices["output"] is not None or devices["input"] is not None

    def __repr__(self) -> str:
        return f"AudioDeviceSwitcher(app_name={self.app_name!r}, process_name={self.process_name!r})"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="GTAV 音频设备切换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m gtaz.audios.cables -l    # 查看当前音频设备
  python -m gtaz.audios.cables -c    # 设置为 CABLE 虚拟音频设备
  python -m gtaz.audios.cables -d    # 设置为系统默认音频设备
        """,
    )
    parser.add_argument(
        "-c",
        "--cable",
        action="store_true",
        help="设置为 CABLE 虚拟音频设备（输出: CABLE Input，输入: CABLE Output）",
    )
    parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="设置为系统默认音频设备",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="查看当前的音频输入输出设备",
    )
    return parser.parse_args()


def main():
    """主函数，处理命令行参数"""
    if len(sys.argv) == 1:
        # 没有参数时显示帮助
        sys.argv.append("-h")
    args = parse_args()

    switcher = AudioDeviceSwitcher()

    if args.list:
        switcher.list_current_devices()

    if args.cable:
        switcher.set_cable()

    if args.default:
        switcher.set_default()


if __name__ == "__main__":
    main()

    # 显示帮助信息
    # python -m gtaz.audios.cables -h

    # 查看当前音频设备
    # python -m gtaz.audios.cables -l

    # 设置为 CABLE 虚拟音频设备
    # python -m gtaz.audios.cables -c

    # 设置为系统默认音频设备
    # python -m gtaz.audios.cables -d
