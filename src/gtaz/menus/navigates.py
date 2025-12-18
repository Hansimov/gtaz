"""GTAV 菜单导航模块"""

from tclogger import TCLogger

from .interacts import MenuInteractor
from .locates import MenuLocatorRunner
from ..screens import ScreenCapturer

logger = TCLogger(name="GTAVMenuNavigator", use_prefix=True, use_prefix_ms=True)


class MenuNavigator:
    def __init__(self):
        self.interactor = MenuInteractor()
        self.locator = MenuLocatorRunner()
        self.capturer = ScreenCapturer()

    def get_menu_names(self) -> list[str]:
        """获取当前菜单项名称列表"""
        frame = self.capturer.capture_frame()
        merged_result = self.locator.locate(frame.to_np())
        return merged_result


def test_menu_navigator():
    logger.note("测试: MenuNavigator...")
    navigator = MenuNavigator()
    navigator.get_menu_names()


if __name__ == "__main__":
    test_menu_navigator()

    # python -m gtaz.menus.navigates
