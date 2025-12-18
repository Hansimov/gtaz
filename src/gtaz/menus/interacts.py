"""GTAV 菜单交互模块"""

from time import sleep

from tclogger import TCLogger

from ..gamepads import GamepadSimulator, Button, sleep_ms


logger = TCLogger(name="GTAVMenuInteractor", use_prefix=True, use_prefix_ms=True)


class MenuInteractor:
    """GTAV 菜单/界面交互"""

    def __init__(self, gamepad: GamepadSimulator = None):
        self.gamepad = gamepad or GamepadSimulator()

    def wait_until_ready(self, duration_ms: int = 100):
        """等待以确保操作生效"""
        sleep_ms(duration_ms)

    def wait_except_first_time(self, duration_ms: int = 100, i: int = 0):
        """等待以确保操作生效，首次操作不等待"""
        if i > 0:
            sleep_ms(duration_ms)

    def _actions_loop(self, times: int, duration_ms: int = 100):
        """在多次循环操作前等待（首次不等待）"""
        for i in range(times):
            self.wait_except_first_time(duration_ms, i)
            yield

    def click_button(self, button: Button, times: int = 1):
        """点击指定按钮"""
        for _ in self._actions_loop(times):
            self.gamepad.click_button(button)

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
        self.click_button(Button.B, times)

    def navigate_up(self, times: int = 1) -> None:
        """菜单向上导航"""
        self.click_button(Button.DPAD_UP, times)

    def navigate_down(self, times: int = 1) -> None:
        """菜单向下导航"""
        self.click_button(Button.DPAD_DOWN, times)

    def navigate_left(self, times: int = 1) -> None:
        """菜单向左导航"""
        self.click_button(Button.DPAD_LEFT, times)

    def navigate_right(self, times: int = 1) -> None:
        """菜单向右导航"""
        self.click_button(Button.DPAD_RIGHT, times)

    def tab_left(self, times: int = 1) -> None:
        """切换到左侧标签页（LB 键）"""
        self.click_button(Button.LEFT_SHOULDER, times)

    def tab_right(self, times: int = 1) -> None:
        """切换到右侧标签页（RB 键）"""
        self.click_button(Button.RIGHT_SHOULDER, times)

    def open_phone(self) -> None:
        """打开手机（上方向键）"""
        self.click_button(Button.DPAD_UP)

    def close_phone(self) -> None:
        """关闭手机（B 键）"""
        self.click_button(Button.B)


class MenuInteractorTester:
    """菜单交互测试"""

    def test(self):
        menu = MenuInteractor()
        logger.note("测试：重置菜单")
        menu.cancel(3)
        sleep(1)

        logger.note("测试：打开暂停菜单 ...")
        menu.open_pause_menu()
        sleep(3)

        logger.note("测试：向右选择标签 x3 ...")
        menu.tab_right(3)
        sleep(2)

        logger.note("测试：向左选择标签 x2 ...")
        menu.tab_left(2)
        sleep(2)

        logger.note("测试：聚焦菜单 ...")
        menu.confirm()
        sleep(2)

        logger.note("测试：向下选择标签 x3 ...")
        menu.navigate_down(3)
        sleep(2)

        logger.note("测试：关闭暂停菜单 ...")
        menu.open_pause_menu()
        # menu.close_menu()
        sleep(1)

        logger.note("测试：打开手机 ...")
        menu.open_phone()
        sleep(3)
        logger.note("测试：关闭手机 ...")
        menu.close_phone()
        sleep(0.5)


def test_menu_interactor():
    tester = MenuInteractorTester()
    tester.test()


if __name__ == "__main__":
    test_menu_interactor()

    # python -m gtaz.menus.interacts
