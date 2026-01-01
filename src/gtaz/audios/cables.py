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
import re
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

    def _log_app_name(self):
        """打印应用名称日志。"""
        logger.mesg(f"应用名称: {self.app_str}")

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
            self._log_device_setting("设置音频输出", device_name)
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
            self._log_device_setting("设置音频输入", device_name)
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
        self._log_title("设置CABLE")
        self._log_app_name()

        output_success = self.set_output_device(CABLE_INPUT_DEVICE)
        input_success = self.set_input_device(CABLE_OUTPUT_DEVICE)

        if output_success and input_success:
            logger.okay("音频设备设置完成：CABLE")
        else:
            logger.warn("部分设备设置失败")

        return output_success and input_success

    def set_default(self) -> bool:
        """
        设置为系统默认音频设备。

        :return: 是否全部设置成功
        """
        self._log_title("设置默认")
        self._log_app_name()

        output_success = self.set_output_device(DEFAULT_RENDER_DEVICE)
        input_success = self.set_input_device(DEFAULT_CAPTURE_DEVICE)

        if output_success and input_success:
            logger.okay("音频设备设置完成：默认")
        else:
            logger.warn("部分设备设置失败")

        return output_success and input_success

    def get_current_devices(self) -> dict:
        """
        获取当前应用使用的音频输入/输出设备信息。

        :return: 包含 'output' 和 'input' 键的字典，值为包含 device_name、friendly_id 和 short_name 的字典
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
            rows = list(csv_reader)

            # 第一步：构建 GUID -> 设备短名称映射
            guid_to_name = {}
            for row in rows:
                if row.get("Type") == "Device":
                    item_id = row.get("Item ID", "")
                    # 提取 GUID：{xxx}.{GUID}
                    match = re.search(r"\{[^}]+\}\.\{([^}]+)\}", item_id)
                    if match:
                        guid = match.group(1)
                        guid_to_name[guid] = row.get("Name", "")

            # 第二步：查找应用的音频设备
            for row in rows:
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
                    friendly_id = row.get("Command-Line Friendly ID", "")
                    item_id = row.get("Item ID", "")

                    # 通过 Item ID 中的 GUID 查找设备短名称
                    short_name = device_name  # 默认使用设备名
                    match = re.search(r"\{[^}]+\}\.\{([^}]+)\}", item_id)
                    if match:
                        guid = match.group(1)
                        short_name = guid_to_name.get(guid, device_name)

                    device_info = {
                        "device_name": device_name,
                        "friendly_id": friendly_id,
                        "short_name": short_name,
                    }

                    if direction == "Render":
                        result["output"] = device_info
                    elif direction == "Capture":
                        result["input"] = device_info

        except Exception as e:
            logger.err(f"解析音频设备信息时出错: {e}")

        return result

    def _format_device_display_name(self, device_info: dict) -> str:
        """
        格式化设备显示名称。

        :param device_info: 包含 device_name、friendly_id 和 short_name 的字典
        :return: 格式化后的显示名称

        格式：设备简称 (实际设备名)
        """
        if not device_info:
            return None

        device_name = device_info.get("device_name", "")
        short_name = device_info.get("short_name", device_name)

        if short_name and short_name != device_name:
            # 有简称且与设备名不同时，显示为：简称 (设备名)
            return f"{short_name} ({device_name})"
        else:
            # 没有简称或简称就是设备名时，只显示设备名
            return device_name

    def list_current_devices(self) -> bool:
        """
        列出应用当前的音频输入输出设备。

        :return: 是否成功获取设备信息
        """
        self._log_title("查看当前")
        self._log_app_name()

        devices = self.get_current_devices()

        # 显示输出设备
        if devices["output"] is not None:
            display_name = self._format_device_display_name(devices["output"])
            self._log_device_setting("当前音频输出", display_name)
        else:
            logger.warn("未找到活动的音频输出设备")

        # 显示输入设备
        if devices["input"] is not None:
            display_name = self._format_device_display_name(devices["input"])
            self._log_device_setting("当前音频输入", display_name)
        else:
            logger.warn("未找到活动的音频输入设备")

        return devices["output"] is not None or devices["input"] is not None

    def _is_cable_input(self, short_name: str) -> bool:
        return short_name == CABLE_INPUT_DEVICE

    def _log_short_name(self, short_name: str):
        if self._is_cable_input(short_name):
            short_name_str = logstr.okay(short_name)
        else:
            short_name_str = logstr.file(short_name)
        logger.mesg(f"当前音频输出: {short_name_str}")

    def _log_okay_setting(self):
        logger.okay("音频输出设备已正确设置")

    def ensure_cable_input(self) -> bool:
        """确保音频输出设备为 CABLE Input

        流程：
        - 检查当前音频输出设备
        - 如果不是 CABLE Input，则设置为 CABLE Input
        - 再次检查音频输出设备
        - 如果仍然不是 CABLE Input，抛出异常并退出

        :return: 是否成功设置
        """
        self._log_title("检查")

        # 第一次检查当前设备
        devices = self.get_current_devices()
        output_device = devices.get("output")

        if output_device is None:
            logger.fail("无法获取当前音频输出设备信息")
            logger.fail("请确保 GTAV 增强版正在运行")
            sys.exit(1)

        short_name = output_device.get("short_name", "")
        self._log_short_name(short_name)

        # 检查是否已经是 CABLE Input
        if self._is_cable_input(short_name):
            self._log_okay_setting()
            return True

        # 不是 CABLE Input，需要设置
        self.set_cable()

        # 第二次检查
        logger.note("检查音频输出设备是否正确设置...")
        devices = self.get_current_devices()
        output_device = devices.get("output")

        if output_device is None:
            logger.fail("设置后无法获取音频输出设备信息")
            logger.fail("请检查音频设备配置")
            sys.exit(1)

        short_name = output_device.get("short_name", "")
        self._log_short_name(short_name)

        # 验证是否设置成功
        if not self._is_cable_input(short_name):
            logger.fail(
                f"音频输出设备设置失败，"
                f"期望: {logstr.file(CABLE_INPUT_DEVICE)}，"
                f"实际: {logstr.file(short_name)}"
            )
            logger.fail("请手动在系统混音器选项中设置 GTAV 的音频输出为 CABLE Input")
            sys.exit(1)

        self._log_okay_setting()
        return True

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
  python -m gtaz.audios.cables -e    # 确保音频输出为 CABLE Input
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
    parser.add_argument(
        "-e",
        "--ensure-cable-input",
        action="store_true",
        help="确保音频输出设备为 CABLE Input",
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

    if args.ensure_cable_input:
        switcher.ensure_cable_input()


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

    # 确保音频输出设备为 CABLE Input
    # python -m gtaz.audios.cables -e
