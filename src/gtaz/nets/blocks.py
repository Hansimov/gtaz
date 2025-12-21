"""GTAV 网络防火墙阻断"""

import argparse
import subprocess
import psutil
import sys

from tclogger import TCLogger, logstr, brk
from typing import Optional


logger = TCLogger(name="FirewallBlocker", use_prefix=True, use_prefix_ms=True)


def run_command(
    cmd_str: str, show_cmd: bool = True, capture_output: bool = True
) -> tuple[bool, str, str]:
    """
    执行命令行命令。

    :param cmd_str: 命令字符串
    :param show_cmd: 是否在日志中显示命令
    :param capture_output: 是否捕获输出

    :return: (是否成功, stdout, stderr)
    """
    try:
        if show_cmd:
            logger.mesg("执行命令:")
            logger.file(cmd_str)
        result = subprocess.run(
            cmd_str,
            capture_output=capture_output,
            text=True,
            encoding="gbk",
            errors="ignore",
            shell=True,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        return result.returncode == 0, stdout, stderr

    except Exception as e:
        logger.err(f"执行命令时出错: {e}")
        return False, "", str(e)


# GTAV 增强版进程名
GTAV_PROCESS_NAME = "GTA5_Enhanced.exe"
# 防火墙规则名称
GTAV_FIREWALL_RULE_NAME = "GTAV_Block_Outbound"


class GTAVFirewallBlocker:
    """
    用于管理 GTAV 增强版游戏的防火墙阻断规则。

    提供添加、删除、启用、禁用防火墙规则的功能。
    """

    def __init__(
        self,
        process_name: str = GTAV_PROCESS_NAME,
        rule_name: str = GTAV_FIREWALL_RULE_NAME,
    ):
        """
        初始化防火墙阻断器。

        :param process_name: 进程名称
        :param rule_name: 防火墙规则名称
        """
        self.process_name = process_name
        self.rule_name = rule_name
        self._process_path: Optional[str] = None

    def find_process_path(self) -> Optional[str]:
        """
        查找 GTAV 进程的完整路径。

        :return: 进程的完整路径，未找到则返回 None
        """
        try:
            for proc in psutil.process_iter(["name", "exe"]):
                if proc.info["name"] == self.process_name:
                    exe_path = proc.info["exe"]
                    if exe_path:
                        self._process_path = exe_path
                        logger.okay(f"GTAV 完整路径:")
                        logger.file(exe_path)
                        return exe_path
            logger.warn(f"未找到运行中的 GTAV 进程: {self.process_name}")
            return None
        except Exception as e:
            logger.err(f"查找进程路径时出错: {e}")
            return None

    @property
    def process_path(self) -> Optional[str]:
        """获取缓存的进程路径，如果未缓存则重新查找。"""
        if self._process_path:
            return self._process_path
        return self.find_process_path()

    def _run_netsh_command(self, cmd_args: str, desc: str) -> bool:
        """
        执行 netsh advfirewall 命令。

        :param cmd_args: 命令参数字符串
        :param desc: 操作描述（用于日志）
        :return: 是否执行成功
        """
        cmd_str = f"netsh advfirewall firewall {cmd_args}"
        success, stdout, stderr = run_command(cmd_str, show_cmd=True)
        if success:
            logger.note(f"更新防火墙规则成功")
            logger.mesg(f"最新状态: {desc}")
            return True
        else:
            logger.warn(f"未能{desc}防火墙规则")
            if stderr:
                logger.warn(f"错误信息:")
                logger.warn(stderr)
            return False

    @property
    def rule_str(self) -> str:
        """获取防火墙规则的日志字符串表示。"""
        return logstr.file(brk(self.rule_name))

    def rule_exists(self) -> bool:
        """
        检查防火墙规则是否存在。

        :return: 规则是否存在
        """
        cmd_str = f"netsh advfirewall firewall show rule name={self.rule_name}"
        success, stdout, stderr = run_command(cmd_str, show_cmd=False)
        if success:
            logger.mesg(f"防火墙规则已存在: {self.rule_str} ")
        else:
            logger.mesg(f"防火墙规则不存在: {self.rule_str} ")
        return success

    def is_rule_enabled(self) -> Optional[bool]:
        """
        检查防火墙规则是否启用。

        :return: 规则是否启用，None 表示规则不存在或检查失败
        """
        cmd_str = f"netsh advfirewall firewall show rule name={self.rule_name}"
        success, stdout, stderr = run_command(cmd_str, show_cmd=False)
        rule_str = logstr.file(brk(self.rule_name))
        if not success:
            logger.warn(f"防火墙规则不存在: {rule_str}")
            return None
        for line in stdout.split("\n"):
            if "Enabled:" in line or "已启用:" in line:
                enabled = "Yes" in line or "是" in line
                if enabled:
                    status_str = logstr.okay("启用")
                else:
                    status_str = logstr.warn("禁用")
                logger.mesg(f"当前状态: {status_str}")
                return enabled
        logger.warn(f"无法确定规则启用状态: {rule_str}")
        return None

    def add_rule(self, path: Optional[str] = None) -> bool:
        """
        添加防火墙规则以阻断 GTAV 的所有出站网络流量。

        :param path: 程序路径，如果为 None 则自动查找
        :return: 是否成功添加规则
        """
        logger.note("=" * 50)
        logger.note("添加防火墙规则")
        logger.note("=" * 50)
        # 检查规则是否已存在
        if self.rule_exists():
            # logger.mesg(f"防火墙规则已存在，无需添加")
            return True
        # 获取程序路径
        path = path or self.process_path
        if path is None:
            logger.fail("无法获取 GTAV 进程路径，请确保游戏正在运行")
            return False
        # 添加阻断出站流量的规则
        cmd_args = f'add rule name={self.rule_name} dir=out action=block program="{path}" enable=yes'
        return self._run_netsh_command(cmd_args, logstr.okay("已添加"))

    def delete_rule(self) -> bool:
        """
        删除防火墙规则。

        :return: 是否成功删除规则
        """
        logger.note("=" * 50)
        logger.note("删除防火墙规则")
        logger.note("=" * 50)
        # 检查规则是否存在
        if not self.rule_exists():
            # logger.mesg(f"防火墙规则不存在，无需删除")
            return True
        cmd_args = f"delete rule name={self.rule_name}"
        return self._run_netsh_command(cmd_args, logstr.warn("已删除"))

    def enable_rule(self) -> bool:
        """
        启用防火墙规则。

        :return: 是否成功启用规则
        """
        logger.note("=" * 50)
        logger.note("启用防火墙规则")
        logger.note("=" * 50)
        # 检查规则是否存在
        if not self.rule_exists():
            logger.warn(f"防火墙规则不存在，无法启用")
            return False
        # 检查规则是否已启用
        if self.is_rule_enabled():
            # logger.mesg(f"防火墙规则已启用，无需重复操作")
            return True
        cmd_args = f"set rule name={self.rule_name} new enable=yes"
        return self._run_netsh_command(cmd_args, logstr.okay("已启用"))

    def disable_rule(self) -> bool:
        """
        禁用防火墙规则。

        :return: 是否成功禁用规则
        """
        logger.note("=" * 50)
        logger.note("禁用防火墙规则")
        logger.note("=" * 50)
        # 检查规则是否存在
        if not self.rule_exists():
            logger.warn(f"防火墙规则不存在，无法禁用")
            return False
        # 检查规则是否已禁用
        enabled = self.is_rule_enabled()
        if enabled is False:
            # logger.mesg(f"防火墙规则已禁用，无需重复操作")
            return True
        cmd_args = f"set rule name={self.rule_name} new enable=no"
        return self._run_netsh_command(cmd_args, logstr.warn("已禁用"))

    def get_rule_info(self) -> Optional[str]:
        """
        获取防火墙规则的详细信息。

        :return: 规则信息字符串，规则不存在则返回 None
        """
        cmd_str = f"netsh advfirewall firewall show rule name={self.rule_name}"
        success, stdout, stderr = run_command(cmd_str, show_cmd=False)
        if success:
            if stdout:
                logger.mesg(f"规则详细信息:")
                logger.mesg(stdout)
            else:
                logger.warn("无法获取规则信息")
            return stdout
        else:
            logger.mesg(f"防火墙规则不存在: {self.rule_str}")
            return None

    def __repr__(self) -> str:
        exists = self.rule_exists()
        enabled = self.is_rule_enabled() if exists else None
        return (
            f"GTAVFirewallBlocker("
            f"rule_name={self.rule_name}, "
            f"process_name={self.process_name}, "
            f"process_path={self._process_path}, "
            f"exists={exists}, "
            f"enabled={enabled})"
        )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="GTAV 防火墙阻断管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m gtaz.nets.blocks -f    # 查找 GTAV 进程路径
  python -m gtaz.nets.blocks -x    # 检查规则是否存在
  python -m gtaz.nets.blocks -a    # 添加防火墙规则
  python -m gtaz.nets.blocks -e    # 启用防火墙规则
  python -m gtaz.nets.blocks -s    # 禁用防火墙规则
  python -m gtaz.nets.blocks -d    # 删除防火墙规则
  python -m gtaz.nets.blocks -i    # 显示规则详细信息
        """,
    )

    parser.add_argument(
        "-x",
        "--exists",
        action="store_true",
        help="检查防火墙规则是否存在",
    )
    parser.add_argument(
        "-a",
        "--add",
        action="store_true",
        help="添加防火墙规则（需要管理员权限）",
    )
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        help="删除防火墙规则（需要管理员权限）",
    )
    parser.add_argument(
        "-e",
        "--enable",
        action="store_true",
        help="启用防火墙规则（需要管理员权限）",
    )
    parser.add_argument(
        "-s",
        "--disable",
        action="store_true",
        help="禁用防火墙规则（需要管理员权限）",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="显示防火墙规则详细信息",
    )
    parser.add_argument(
        "-f",
        "--find",
        action="store_true",
        help="查找并打印 GTAV 进程的完整路径",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="手动指定 GTAV 进程路径（用于添加规则）",
    )
    return parser.parse_args()


def main():
    """主函数，处理命令行参数"""
    if len(sys.argv) == 1:
        # 没有参数时显示帮助
        sys.argv.append("-h")
    args = parse_args()

    blocker = GTAVFirewallBlocker()

    if args.find:
        process_path = blocker.find_process_path()

    if args.exists:
        exists = blocker.rule_exists()

    if args.info:
        rule_info = blocker.get_rule_info()

    if args.add:
        blocker.add_rule(path=args.path)

    if args.enable:
        # blocker.add_rule(path=args.path)
        blocker.enable_rule()

    if args.disable:
        blocker.disable_rule()

    if args.delete:
        blocker.delete_rule()


if __name__ == "__main__":
    main()

    # 显示帮助信息
    # python -m gtaz.nets.blocks

    # 查找 GTAV 进程路径
    # python -m gtaz.nets.blocks -f

    # 检查防火墙规则是否存在
    # python -m gtaz.nets.blocks -x

    # 添加防火墙规则
    # python -m gtaz.nets.blocks -a

    # 启用防火墙规则
    # python -m gtaz.nets.blocks -e

    # 禁用防火墙规则
    # python -m gtaz.nets.blocks -s

    # 删除防火墙规则
    # python -m gtaz.nets.blocks -d

    # 显示规则详细信息
    # python -m gtaz.nets.blocks -i
