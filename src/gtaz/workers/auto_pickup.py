"""GTAV 自动循环取货模块"""

import argparse
import sys

from time import sleep
from tclogger import TCLogger, logstr, Runtimer, dt_to_str

from ..workers.mode_switch import NetmodeSwitcher
from ..nets.blocks import GTAVFirewallBlocker
from ..audios.detects_v2 import AudioDetector


logger = TCLogger(name="AutoPickuper", use_prefix=True, use_prefix_ms=True)


# 默认循环次数
LOOP_COUNT = 1
# 切换到故事模式后的等待时间（秒）
WAIT_AT_STORY = 15
# 确保货物全部到达等待时间（秒）
WAIT_FOR_GOODS = 15
# 等待音频信号稳定时间（秒）
WAIT_FOR_QUIET = 15
# 检测到音频信号后等待时间（秒）
WAIT_AFTER_DETECT = 1
# 等待断网提示出现（秒）
WAIT_FOR_DISCONNECT_WARN = 12
# 线上模式确认次数
WARN_CONFIRM_COUNT = 3
# 相邻确认的间隔（秒）
WARN_CONFIRM_INTERVAL = 1


class AutoPickuper:
    """GTA 自动循环取货器

    整合 NetmodeSwitcher、GTAVFirewallBlocker、AudioDetector，
    实现自动切换模式、监控音频信号、控制防火墙的完整自动循环取货流程。
    """

    def __init__(self):
        """初始化自动循环取货器"""
        self.switcher = NetmodeSwitcher()
        self.blocker = GTAVFirewallBlocker()
        self.detector = AudioDetector()
        self.detector.initialize()

    # =============== 故事模式相关 =============== #
    def _do_at_story(self):
        """故事模式操作"""
        sleep(WAIT_AT_STORY)

    def switch_to_story(self) -> bool:
        """切换到故事模式

        流程：
        - 切换到故事模式
        - 等待指定秒数

        :return: 是否成功完成流程
        """
        # 切换到故事模式
        self.switcher.switch_online_to_story()
        # TODO: 后续优化：检测是否已经切到故事模式
        return True

    # =============== 在线模式相关 =============== #
    def _wait_for_quiet(self):
        """等待音频信号稳定"""
        logger.note(f"等待 {WAIT_FOR_QUIET} 秒，直到音频信号稳定...")
        sleep(WAIT_FOR_QUIET)

    def _wait_for_disconnect_warn(self):
        """等待断网警报"""
        # TODO: 后续优化为检测到加载图像
        logger.note(f"等待 {WAIT_FOR_DISCONNECT_WARN} 秒，以确认断网提示...")
        sleep(WAIT_FOR_DISCONNECT_WARN)

    def _confirm_disconnect_warn(self):
        """确认断网警报"""
        for i in range(WARN_CONFIRM_COUNT):
            self.switcher.interactor.confirm()
            sleep(WARN_CONFIRM_INTERVAL)

    def _wait_for_goods_arrival(self):
        """等待确保货物全部到达"""
        logger.note(f"等待 {WAIT_FOR_GOODS} 秒，以确保货物全部到达...")
        sleep(WAIT_FOR_GOODS)

    def _do_at_online(self):
        """在线模式操作"""
        # 等待断网提示
        self._wait_for_disconnect_warn()
        # 确认断网警报
        self._confirm_disconnect_warn()
        # 等待确保货物全部到达
        self._wait_for_goods_arrival()
        # 再次确认断网警报以防万一
        self._confirm_disconnect_warn()

    def switch_to_invite(self) -> bool:
        """切换到在线模式（邀请战局）

        流程：
        - 禁用防火墙规则
        - 切换到在线模式（邀请战局）
        - 等待音频信号稳定再开启检测
        - 启动音量信号检测，检测到 1 次后立即退出
        - 启用防火墙规则
        - 确认断网提示
        - 等待货物全部到达

        :return: 是否成功完成流程
        """
        # 禁用防火墙规则
        self.blocker.disable_rule()
        sleep(3)
        # 切换到在线模式
        self.switcher.switch_to_new_invite_lobby()
        # 等待音频信号稳定再开启检测
        self._wait_for_quiet()
        # 启动音频检测，检测到匹配信号后立即退出
        matched, score, result = self.detector.detect_then_stop()
        # 检测到信号后，等待一段时间
        sleep(WAIT_AFTER_DETECT)
        # 启用防火墙规则
        self.blocker.enable_rule()
        return True

    # =============== 循环 =============== #
    def _do_at_last_round(self):
        """收尾操作"""
        logger.hint("已完成最后一轮循环，将执行如下操作：")
        logger.hint("[1] 禁用防火墙规则，恢复网络连接")
        logger.hint("[2] 同步存档到服务器")
        self.blocker.disable_rule()
        logger.note("等待 10 秒，确保网络连接恢复...")
        sleep(10)
        self.switcher.sync_to_remote()

    def _log_at_round_start(self, round: int):
        logger.hint("=" * 50)
        logger.hint(f"[{logstr.file(round+1)}/{self.loop_count}] 循环开始")
        logger.hint("=" * 50)

    def _log_at_round_end(self, round: int):
        self.timer.elapsed_time()
        round_str = logstr.file(round + 1)
        elapsed_str = dt_to_str(self.timer.dt, str_format="unit")
        elapsed_str = logstr.mesg(elapsed_str)
        logger.okay(f"第 {round_str} 轮完成，用时: {elapsed_str}")

    def switch_loop(
        self, loop_count: int = LOOP_COUNT, args: argparse.Namespace = None
    ) -> bool:
        """循环切换模式

        流程：循环调用 switch_to_invite 和 switch_to_story，直到指定次数

        :param loop_count: 循环次数
        :return: 是否成功完成所有循环
        """
        # 添加防火墙规则
        self.blocker.add_rule()
        # 打印循环开始信息
        logger.hint(f"开始循环切换，循环次数: {loop_count}")
        self.loop_count = loop_count
        self.timer = Runtimer(verbose=False)
        for i in range(loop_count):
            # 打印本轮开始信息
            self._log_at_round_start(i)
            self.timer.start_time()
            # 确保在故事模式
            self.switch_to_story()
            # 切换到在线模式（邀请战局）
            self.switch_to_invite()
            # 执行在线模式操作
            self._do_at_online()
            # 最后一轮
            if i >= loop_count - 1:
                # 打印最后一轮结束信息
                self.timer.end_time()
                self._log_at_round_end(i)
                # 执行收尾操作
                self._do_at_last_round()
                if args and not args.go_to_story_after_finished:
                    # 最后一轮留在线上，不再回到故事模式
                    break
            # 切换到故事模式
            self.switch_to_story()
            # 执行故事模式操作
            self._do_at_story()
            # 打印本轮结束信息
            self.timer.end_time()
            self._log_at_round_end(i)
        logger.okay("所有循环完成")
        return True

    def __repr__(self) -> str:
        return f"AutoPickuper()"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="GTAV 自动循环取货工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m gtaz.workers.auto_pickup -s          # 切换到故事模式
  python -m gtaz.workers.auto_pickup -i          # 切换到在线模式（邀请战局）
  python -m gtaz.workers.auto_pickup -l          # 循环1次
  python -m gtaz.workers.auto_pickup -l -g       # 循环1次，结束后回到线下
  python -m gtaz.workers.auto_pickup -l -c 5     # 循环5次
        """,
    )
    parser.add_argument(
        "-s",
        "--story",
        action="store_true",
        help="切换到故事模式",
    )
    parser.add_argument(
        "-i",
        "--invite",
        action="store_true",
        help="切换到在线模式（邀请战局）",
    )
    parser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        help="循环切换模式",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=LOOP_COUNT,
        help=f"循环次数（默认: {LOOP_COUNT}）",
    )
    parser.add_argument(
        "-g",
        "--go-to-story-after-finished",
        action="store_true",
        help="循环结束后回到线下故事模式（默认留在线上）",
    )
    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        # 没有参数时显示帮助
        sys.argv.append("-h")
    args = parse_args()

    pickuper = AutoPickuper()

    if args.story:
        pickuper.switch_to_story()

    if args.invite:
        pickuper.switch_to_invite()

    if args.loop:
        pickuper.switch_loop(loop_count=args.count, args=args)


if __name__ == "__main__":
    main()

    # 显示帮助信息
    # python -m gtaz.workers.auto_pickup -h

    # 切换到故事模式
    # python -m gtaz.workers.auto_pickup -s

    # 切换到在线模式（邀请战局）
    # python -m gtaz.workers.auto_pickup -i

    # 循环（默认1次）
    # python -m gtaz.workers.auto_pickup -l

    # 循环1次，结束后回到线下故事模式
    # python -m gtaz.workers.auto_pickup -l -g

    # 循环5次
    # python -m gtaz.workers.auto_pickup -l -c 5
