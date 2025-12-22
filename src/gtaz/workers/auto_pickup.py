"""GTAV 自动循环取货模块"""

import argparse
import sys
import time

from tclogger import TCLogger, logstr

from ..workers.mode_switch import NetmodeSwitcher
from ..nets.blocks import GTAVFirewallBlocker
from ..audios.detects import SignalDetector


logger = TCLogger(name="AutoPickuper", use_prefix=True, use_prefix_ms=True)


# 默认循环次数
LOOP_COUNT = 10
# 切换到故事模式后的等待时间（秒）
SECS_AT_STORY = 5
# 确认断网提示前的等待时间（秒）
SECS_BEFORE_CONFIRM_BLOCK = 5
# 检测到信号后的等待时间（秒）
SECS_AT_ONLINE = 10
# 等待音频信号稳定时间（秒）
SECS_BEFORE_DETECT = 2
# 线上模式确认次数
CONFIRM_COUNT_AT_HINT = 2


class AutoPickuper:
    """GTA 自动循环取货器

    整合 NetmodeSwitcher、GTAVFirewallBlocker、SignalDetector，
    实现自动切换模式、监控音频信号、控制防火墙的完整自动循环取货流程。
    """

    def __init__(
        self,
        secs_at_story: float = SECS_AT_STORY,
        secs_at_online: float = SECS_AT_ONLINE,
        secs_before_detect: float = SECS_BEFORE_DETECT,
        confirm_count_at_hint: int = CONFIRM_COUNT_AT_HINT,
    ):
        """初始化自动循环取货器

        :param secs_at_story: 切换到故事模式后的等待时间（秒）
        :param secs_at_online: 检测到信号后的等待时间（秒）
        :param secs_before_detect: 等待音频信号稳定时间（秒）
        :param confirm_count_at_hint: 确认次数
        """
        self.secs_at_story = secs_at_story
        self.secs_at_online = secs_at_online
        self.secs_before_detect = secs_before_detect
        self.confirm_count_at_hint = confirm_count_at_hint
        # 初始化各个组件
        self.switcher = NetmodeSwitcher()
        self.blocker = GTAVFirewallBlocker()
        self.detector = SignalDetector()

    def _confirm_at_online(self):
        """在线模式确认操作"""
        # TODO: 后续优化为检测到加载图像
        time.sleep(SECS_BEFORE_CONFIRM_BLOCK)
        for i in range(self.confirm_count_at_hint):
            self.switcher.interactor.confirm()

    def _sleep_at_online(self):
        """在线模式等待操作"""
        time.sleep(self.secs_at_online)

    def _sleep_after_disable_rule(self):
        """禁用防火墙规则后的等待操作"""
        time.sleep(3)

    def _sleep_before_detect(self):
        """等待音频信号稳定"""
        time.sleep(self.secs_before_detect)

    def switch_to_online(self) -> bool:
        """切换到在线模式

        流程：
        - 禁用防火墙规则
        - 切换到在线模式
        - 等待音频信号稳定再开启检测
        - 启动音量信号检测，检测到 1 次后立即退出
        - 启用防火墙规则
        - 确认断网提示
        - 等待货物全部到达

        :return: 是否成功完成流程
        """
        # 禁用防火墙规则
        self.blocker.disable_rule()
        self._sleep_after_disable_rule()
        # 切换到在线模式
        self.switcher.switch_story_to_online()
        # 等待音频信号稳定再开启检测
        self._sleep_before_detect()
        # 启动音量信号检测，检测到 2 次后立即退出
        self.detector.stop_after_detect(count=2, interval=0)
        # 启用防火墙规则
        self.blocker.enable_rule()
        # 确认断网提示
        self._confirm_at_online()
        # 等待货物全部到达
        self._sleep_at_online()
        return True

    def _sleep_at_story(self):
        """故事模式等待操作"""
        time.sleep(self.secs_at_story)

    def switch_to_story(self) -> bool:
        """切换到故事模式

        流程：
        - 切换到故事模式
        - 等待指定秒数

        :return: 是否成功完成流程
        """
        # 切换到故事模式
        self.switcher.switch_online_to_story()
        # 等待指定秒数
        self._sleep_at_story()

        return True

    def switch_loop(self, loop_count: int = LOOP_COUNT) -> bool:
        """循环切换模式

        流程：循环调用 switch_to_story + switch_to_online 指定次数

        :param loop_count: 循环次数
        :return: 是否成功完成所有循环
        """
        logger.note("=" * 50)
        logger.note("开始循环切换模式")
        logger.note("=" * 50)
        logger.note(f"循环次数: {loop_count}")
        for i in range(loop_count):
            logger.note("=" * 50)
            logger.hint(f"[{logstr.file(i+1)}/{loop_count}] 循环开始")
            logger.note("=" * 50)
            # 切换到在线模式
            self.switch_to_online()
            # 切换到故事模式
            self.switch_to_story()
        logger.okay("所有循环完成")
        return True

    def __repr__(self) -> str:
        return (
            f"AutoPickuper("
            f"secs_at_story={self.secs_at_story}, "
            f"secs_at_online={self.secs_at_online}, "
            f"confirm_count_at_hint={self.confirm_count_at_hint}"
        )


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
  python -m gtaz.workers.auto_pickup -o          # 切换到在线模式
  python -m gtaz.workers.auto_pickup -l          # 循环切换（默认10次）
  python -m gtaz.workers.auto_pickup -l -c 5     # 循环切换5次
        """,
    )

    parser.add_argument(
        "-s",
        "--story",
        action="store_true",
        help="切换到故事模式",
    )
    parser.add_argument(
        "-o",
        "--online",
        action="store_true",
        help="切换到在线模式",
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
    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        # 没有参数时显示帮助
        sys.argv.append("-h")
    args = parse_args()

    pickuper = AutoPickuper()

    if args.story:
        pickuper.switch_to_story()

    if args.online:
        pickuper.switch_to_online()

    if args.loop:
        pickuper.switch_loop(loop_count=args.count)


if __name__ == "__main__":
    main()

    # 显示帮助信息
    # python -m gtaz.workers.auto_pickup

    # 切换到故事模式
    # python -m gtaz.workers.auto_pickup -s

    # 切换到在线模式
    # python -m gtaz.workers.auto_pickup -o

    # 循环切换（默认10次）
    # python -m gtaz.workers.auto_pickup -l

    # 循环切换5次
    # python -m gtaz.workers.auto_pickup -l -c 5
