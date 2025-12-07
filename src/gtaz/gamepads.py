"""模拟手柄"""

"""
References:
- https://github.com/shibeta/JNTMbot_python/blob/main/gamepad_utils.py
"""

import enum
import time
import atexit
import vgamepad as vg

from tclogger import TCLogger
from typing import Union


logger = TCLogger(name="GamepadSimulator", use_prefix=True)


class Button(enum.IntFlag):
    """手柄按键映射"""

    A = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
    B = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
    X = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
    Y = vg.XUSB_BUTTON.XUSB_GAMEPAD_Y
    DPAD_UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
    DPAD_DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
    DPAD_LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
    DPAD_RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
    CROSS_KEY_UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
    CROSS_KEY_DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
    CROSS_KEY_LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
    CROSS_KEY_RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
    START = vg.XUSB_BUTTON.XUSB_GAMEPAD_START  # 靠右的小按钮
    MENU = vg.XUSB_BUTTON.XUSB_GAMEPAD_START  # START 的别名
    BACK = vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK  # 靠左的小按钮
    SELECT = vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK  # BACK 的别名
    LEFT_STICK = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB
    RIGHT_STICK = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB
    LEFT_SHOULDER = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
    RIGHT_SHOULDER = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER


AnyButton = Union[vg.XUSB_BUTTON, Button]


class JoystickDirection(tuple[float, float]):
    """常用摇杆方向映射"""

    CENTER = (0.0, 0.0)

    HALF_UP = (0.0, 0.7)
    HALF_DOWN = (0.0, -0.7)
    HALF_LEFT = (-0.7, 0.0)
    HALF_RIGHT = (0.7, 0.0)

    HALF_LEFTUP = (-0.6, 0.6)
    HALF_RIGHTUP = (0.6, 0.6)
    HALF_LEFTDOWN = (-0.6, -0.6)
    HALF_RIGHTDOWN = (0.6, -0.6)

    FULL_UP = (0.0, 1.0)
    FULL_DOWN = (0.0, -1.0)
    FULL_LEFT = (-1.0, 0.0)
    FULL_RIGHT = (1.0, 0.0)

    FULL_LEFTUP = (-1.0, 1.0)
    FULL_RIGHTUP = (1.0, 1.0)
    FULL_LEFTDOWN = (-1.0, -1.0)
    FULL_RIGHTDOWN = (1.0, -1.0)


AnyJoystickDirection = tuple[float, float]


class TriggerPressure:
    """常用扳机压力值映射"""

    released = 0.0  # 完全松开
    light = 0.4  # 轻压 (适用于需要精确控制的场景，如半按加速)
    full = 1.0  # 完全按下 (适用于射击等场景)


AnyTriggerPressure = float


class GamepadSimulator:
    """
    用于模拟手柄操作的类。
    在程序退出时，会自动释放所有手柄按键、扳机和摇杆，防止卡住。
    """

    def __init__(self):
        try:
            self.pad = vg.VX360Gamepad()
            logger.okay("虚拟手柄设备已创建。")
            # 注册清理函数
            atexit.register(self._cleanup)
            # 初始化手柄状态
            self.pad.reset()
            # 按一下A键以唤醒手柄
            self.click_button(Button.A)
            logger.okay("初始化虚拟手柄完成。")

        except Exception as e:
            logger.err(
                f"初始化虚拟手柄失败: {e}。请确保已安装 ViGEmBus 驱动，并且没有其他程序正在使用 ViGEmBus 模拟手柄。"
            )
            logger.note("请参考如下文件安装驱动：")
            logger.file(
                "https://github.com/shibeta/JNTMbot_python/blob/main/install_vigembus.bat"
            )
            raise

    def _cleanup(self):
        """程序退出时调用的清理函数。"""
        if self.pad:
            try:
                logger.note("正在重置虚拟手柄状态...")
                self.pad.reset()
                self.pad.update()
                logger.okay("虚拟手柄状态已重置。")
            except Exception as e:
                logger.warn(f"重置虚拟手柄时出错: {e}")

    def _check_connected(self) -> bool:
        if self.pad is None:
            logger.err("没有安装虚拟手柄驱动，或没有初始化")
            return False
        return True

    def press_button(self, button: AnyButton):
        """
        按下一个按钮。

        :param button: 要按下的按钮
        """
        if not self._check_connected():
            return
        try:
            self.pad.press_button(button)
            self.pad.update()
        except Exception as e:
            logger.err(f"按下按钮 {button} 时出错: {e}")

    def release_button(self, button: AnyButton):
        """
        松开一个按钮。

        :param button: 要松开的按钮
        """
        if not self._check_connected():
            return
        try:
            self.pad.release_button(button)
            self.pad.update()
        except Exception as e:
            logger.err(f"松开按钮 {button} 时出错: {e}")

    def click_button(self, button: AnyButton, duration_milliseconds: int = 100):
        """
        按住一个按钮，一段时间后松开。

        :param button: 要按住的按钮
        :param duration_milliseconds: 持续时间，单位为毫秒
        """
        if not self._check_connected():
            return
        try:
            self.press_button(button)
            time.sleep(duration_milliseconds / 1000.0)
        except Exception as e:
            logger.err(f"点按按钮 {button} 时出错: {e}")
        finally:
            self.release_button(button)

    def return_left_joystick_to_center(self):
        self.move_left_joystick(JoystickDirection.CENTER)

    def move_left_joystick(self, direction: tuple[float, float]):
        """
        推动左摇杆到某位置。

        :param direction: 左右方向和后前方向，取值范围为 -1.0 ~ 1.0. (0代表回中)
        """
        if not self._check_connected():
            return
        try:
            self.pad.left_joystick_float(*direction)
            self.pad.update()
        except Exception as e:
            logger.err(f"移动左摇杆时出错: {e}")

    def hold_left_joystick(
        self, direction: tuple[float, float], duration_milliseconds: int = 100
    ):
        """
        推动左摇杆到某位置，一段时间后回中。

        :param direction: 左右方向和后前方向，取值范围为 -1.0 ~ 1.0. (0代表回中)
        :param duration_milliseconds: 持续时间，单位为毫秒
        """
        if not self._check_connected():
            return
        self.move_left_joystick(direction)
        time.sleep(duration_milliseconds / 1000.0)
        self.return_left_joystick_to_center()
        self.pad.update()

    def return_right_joystick_to_center(self):
        self.move_right_joystick(JoystickDirection.CENTER)

    def move_right_joystick(self, direction: tuple[float, float]):
        """
        推动右摇杆到某位置。

        :param direction: 左右方向和后前方向，取值范围为 -1.0 ~ 1.0. (0代表回中)
        """
        if not self._check_connected():
            return
        try:
            self.pad.right_joystick_float(*direction)
            self.pad.update()
        except Exception as e:
            logger.err(f"Error moving right stick: {e}")

    def hold_right_joystick(
        self, direction: tuple[float, float], duration_milliseconds: int = 100
    ):
        """
        推动右摇杆到某位置，一段时间后回中。

        :param direction: 左右方向和后前方向，取值范围为 -1.0 ~ 1.0. (0代表回中)
        :param duration_milliseconds: 持续时间，单位为毫秒
        """
        if not self._check_connected():
            return
        self.move_right_joystick(direction)
        time.sleep(duration_milliseconds / 1000.0)
        self.return_right_joystick_to_center()
        self.pad.update()

    def press_left_trigger(self, pressure_float: float):
        """
        按压左扳机到指定压力值。

        :param pressure_float: 压力值，取值范围为 0.0 (松开) ~ 1.0 (完全按下)。
        """
        if not self._check_connected():
            return
        try:
            self.pad.left_trigger_float(value_float=pressure_float)
            self.pad.update()
        except Exception as e:
            logger.err(f"按压左扳机时出错: {e}")

    def release_left_trigger(self):
        """完全松开左扳机。"""
        self.press_left_trigger(TriggerPressure.released)

    def hold_left_trigger(
        self, pressure_float: float, duration_milliseconds: int = 100
    ):
        """
        按住左扳机一段时间后松开。

        :param pressure_float: 压力值，取值范围为 0.0 (松开) ~ 1.0 (完全按下)。
        :param duration_milliseconds: 持续时间，单位为毫秒。
        """
        if not self._check_connected():
            return
        self.press_left_trigger(pressure_float)
        time.sleep(duration_milliseconds / 1000.0)
        self.release_left_trigger()

    def press_right_trigger(self, pressure_float: float):
        """
        按压右扳机到指定压力值。

        :param pressure_float: 压力值，取值范围为 0.0 (松开) ~ 1.0 (完全按下)。
        """
        if not self._check_connected():
            return
        try:
            self.pad.right_trigger_float(value_float=pressure_float)
            self.pad.update()
        except Exception as e:
            logger.err(f"按压右扳机时出错: {e}")

    def release_right_trigger(self):
        """完全松开右扳机。"""
        self.press_right_trigger(TriggerPressure.released)

    def hold_right_trigger(
        self, pressure_float: float, duration_milliseconds: int = 100
    ):
        """
        按住右扳机一段时间后松开。

        :param pressure_float: 压力值，取值范围为 0.0 (松开) ~ 1.0 (完全按下)。
        :param duration_milliseconds: 持续时间，单位为毫秒。
        """
        if not self._check_connected():
            return
        self.press_right_trigger(pressure_float)
        time.sleep(duration_milliseconds / 1000.0)
        self.release_right_trigger()


def test_gamepad_simulator():
    simulator = GamepadSimulator()
    simulator.click_button(Button.A)
    simulator.hold_left_joystick(JoystickDirection.FULL_UP, duration_milliseconds=500)
    simulator.hold_right_trigger(TriggerPressure.full, duration_milliseconds=500)


if __name__ == "__main__":
    test_gamepad_simulator()

    # python -m gtaz.gamepads
