"""GTAV 菜单交互模块"""

import time

from tclogger import TCLogger

from ..gamepads import GamepadSimulator, Button, sleep_ms


logger = TCLogger(name="GTAVMenuInteractor", use_prefix=True, use_prefix_ms=True)


class MenuInteractor:
    """GTAV 菜单/界面交互"""

    def __init__(self, gamepad: GamepadSimulator = None):
        self.gamepad = gamepad or GamepadSimulator()

    def wait_until_ready(self, duration_ms: int = 200):
        """等待以确保操作生效"""
        sleep_ms(duration_ms)

    def click_button(self, button: Button):
        """点击指定按钮"""
        self.gamepad.click_button(button)
        self.wait_until_ready()

    def open_pause_menu(self) -> None:
        """打开暂停菜单（START 键）"""
        self.click_button(Button.START)

    def hide_menu(self) -> None:
        """隐藏菜单（Y 键）"""
        self.click_button(Button.Y)

    def close_menu(self) -> None:
        """关闭当前菜单（B 键）"""
        self.click_button(Button.B)

    def open_interaction_menu(self) -> None:
        """打开互动菜单（长按 BACK/SELECT 键）"""
        self.gamepad.press_button(Button.BACK, 1000)
        self.wait_until_ready()

    def confirm(self) -> None:
        """确认/选择（A 键）"""
        self.click_button(Button.A)

    def cancel(self, times: int = 1) -> None:
        """取消/返回（B 键）"""
        for _ in range(times):
            self.click_button(Button.B)

    def navigate_up(self) -> None:
        """菜单向上导航"""
        self.click_button(Button.DPAD_UP)

    def navigate_down(self) -> None:
        """菜单向下导航"""
        self.click_button(Button.DPAD_DOWN)

    def navigate_left(self) -> None:
        """菜单向左导航"""
        self.click_button(Button.DPAD_LEFT)

    def navigate_right(self) -> None:
        """菜单向右导航"""
        self.click_button(Button.DPAD_RIGHT)

    def tab_left(self) -> None:
        """切换到左侧标签页（LB 键）"""
        self.click_button(Button.LEFT_SHOULDER)

    def tab_right(self) -> None:
        """切换到右侧标签页（RB 键）"""
        self.click_button(Button.RIGHT_SHOULDER)

    def open_phone(self) -> None:
        """打开手机（上方向键）"""
        self.click_button(Button.DPAD_UP)

    def close_phone(self) -> None:
        """关闭手机（B 键）"""
        self.click_button(Button.B)


class MenuNavigator:
    pass


class MenuInteractorTester:
    """菜单交互测试"""

    def test(self):
        menu = MenuInteractor()
        logger.note("测试：重置菜单")
        menu.cancel(3)
        time.sleep(1)

        logger.note("测试：打开暂停菜单 ...")
        menu.open_pause_menu()
        time.sleep(3)
        logger.note("测试：关闭暂停菜单 ...")
        menu.close_menu()
        time.sleep(1)

        logger.note("测试：打开手机 ...")
        menu.open_phone()
        time.sleep(3)
        logger.note("测试：关闭手机 ...")
        menu.close_phone()
        time.sleep(0.5)


def test_menu_interactor():
    tester = MenuInteractorTester()
    tester.test()


if __name__ == "__main__":
    test_menu_interactor()

    # python -m gtaz.menus.interacts
