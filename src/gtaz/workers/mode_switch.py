"""GTA 在线/故事 模式切换模块"""

from tclogger import TCLogger, logstr

from ..menus.navigates import MenuNavigator
from ..menus.locates import ExitLocatorRunner, is_score_too_low
from ..screens import ScreenCapturer


logger = TCLogger(name="NetmodeSwitcher", use_prefix=True, use_prefix_ms=True)


class NetmodeSwitcher:
    """GTA 在线/故事模式切换器"""

    def __init__(self):
        """初始化模式切换器"""
        self.navigator = MenuNavigator()
        self.capturer = ScreenCapturer()
        # 使用 navigator 中的 interactor，避免创建新的手柄（会不生效）
        self.interactor = self.navigator.interactor
        # 使用 navigator 中的 locator
        self.locator = self.navigator.locator_runner.locator
        self.exit_runner = ExitLocatorRunner()

    def get_netmode(self) -> str:
        """获取当前模式

        :return: 当前模式名称，"在线模式" 或 "故事模式"，如果无法识别则返回 None
        """
        # 确保菜单打开
        self.navigator.ensure_menu_opened()
        # 截取屏幕
        frame_np = self.capturer.capture_frame(verbose=False).to_np()
        # 使用 navigator 的 locator 来匹配模式
        result = self.navigator.locator_runner.locator.match_mode(frame_np)
        # 判断匹配结果
        if not is_score_too_low(result):
            mode_name = result.name
            logger.mesg(f"当前模式: [{mode_name}]")
            return mode_name
        else:
            logger.warn("无法识别当前模式")
            return None

    def exit_and_confirm(self, max_retries: int = 10) -> bool:
        """定位退出提示并确认

        :param max_retries: 最大重试次数
        :return: 是否成功完成定位-确认流程
        """
        logger.note("定位退出提示并确认...")

        retry = 0
        while retry < max_retries:
            retry += 1
            # 等待退出提示出现
            self.interactor.wait_until_ready(500)
            # 截取屏幕
            frame_np = self.capturer.capture_frame(verbose=False).to_np()
            # 定位退出提示
            exit_result = self.exit_runner.locate(frame_np, verbose=True)
            # 判断是否匹配到退出提示
            if not is_score_too_low(exit_result):
                logger.okay(f"已定位到退出提示: {exit_result.name}")
                # 确认退出
                self.interactor.confirm()
                logger.okay("已确认退出")
                return True
            else:
                retry_str = f"[{retry} / {max_retries}]"
                logger.warn(f"{retry_str} 未定位到退出提示，重试中...")
                # 再按一次 confirm，可能需要多次确认才能触发退出提示
                self.interactor.confirm()
        logger.fail("未能定位到退出提示")
        return False

    def _switch_mode(
        self, dst_names: list[str], mode_desc: str, max_retries: int = 5
    ) -> bool:
        """通用模式切换方法

        :param dst_names: 目标菜单路径
        :param mode_desc: 模式切换描述（用于日志）
        :param max_retries: 最大重试次数
        :return: 是否成功切换模式
        """
        logger.note("=" * 50)
        logger.note(mode_desc)
        logger.note("=" * 50)

        retry = 0
        while retry < max_retries:
            retry += 1
            if retry > 1:
                retry_str = f"[{retry} / {max_retries}] "
            else:
                retry_str = ""
            logger.note(f"{retry_str}尝试导航到: {dst_names}")
            # 导航到目标菜单项
            current_names = self.navigator.go_to(dst_names)
            # 检查是否导航到目标位置
            if current_names == dst_names:
                logger.okay(f"已导航到目标位置: {current_names}")
                # 确认选择
                self.interactor.confirm()
                logger.mesg("已确认选择，等待退出提示...")
                # 定位退出提示并确认
                if self.exit_and_confirm():
                    logger.okay("已完成切换")
                    return True
                # 如果定位退出提示失败，继续重试整个流程
            else:
                logger.warn(f"未导航到目标位置，当前位置: {current_names}，重试...")

        logger.fail("模式切换失败")
        return False

    def _log_okay_mode(self, mode_name: str) -> None:
        """记录当前模式

        :param mode_name: 当前模式名称
        """
        logger.note(f"当前已是{logstr.okay(mode_name)}，无需切换")

    def switch_story_to_online(self, max_retries: int = 5) -> bool:
        """故事模式切换到在线模式

        :param max_retries: 最大重试次数
        :return: 是否成功切换模式
        """
        # 检查当前模式
        mode = self.get_netmode()
        if mode == "在线模式":
            self._log_okay_mode(mode)
            return True

        dst_names = ["在线", "进入GTA在线模式", "凭邀请加入的战局"]
        return self._switch_mode(dst_names, "故事模式 -> 在线模式", max_retries)

    def switch_online_to_story(self, max_retries: int = 5) -> bool:
        """在线模式切换到故事模式

        :param max_retries: 最大重试次数
        :return: 是否成功切换模式
        """
        # 检查当前模式
        mode = self.get_netmode()
        if mode == "故事模式":
            self._log_okay_mode(mode)
            return True

        dst_names = ["在线", "退至故事模式"]
        return self._switch_mode(dst_names, "在线模式 -> 故事模式", max_retries)

    def switch_to_new_invite_lobby(self, max_retries: int = 5) -> bool:
        """切换到新的邀请战局

        根据当前模式自动选择对应的菜单路径：
        - 故事模式: 在线 -> 进入GTA在线模式 -> 凭邀请加入的战局
        - 在线模式: 在线 -> 寻找新战局 -> 仅限邀请的战局

        :param max_retries: 最大重试次数
        :return: 是否成功切换到新的邀请战局
        """
        # 获取当前模式
        mode = self.get_netmode()
        if not mode:
            logger.fail("无法识别当前模式，切换失败")
            return False

        # 根据模式设置目标路径和描述
        if mode == "故事模式":
            dst_names = ["在线", "进入GTA在线模式", "凭邀请加入的战局"]
            mode_desc = "切换到新的邀请战局（从故事模式）"
        else:  # mode == "在线模式"
            dst_names = ["在线", "寻找新战局", "仅限邀请的战局"]
            mode_desc = "切换到新的邀请战局（从在线模式）"

        # 执行切换
        return self._switch_mode(dst_names, mode_desc, max_retries)


def test_netmode_switcher():
    """测试模式切换器"""
    logger.note("测试: GTAVNetmodeSwitcher...")
    switcher = NetmodeSwitcher()

    # 故事模式 -> 在线模式
    # switcher.switch_story_to_online()

    # 在线模式 -> 故事模式
    # switcher.switch_online_to_story()

    # 新的邀请战局
    switcher.switch_to_new_invite_lobby()


if __name__ == "__main__":
    test_netmode_switcher()

    # python -m gtaz.workers.mode_switch
