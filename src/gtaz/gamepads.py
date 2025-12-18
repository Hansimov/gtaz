"""模拟手柄"""

"""
References:
- https://github.com/shibeta/JNTMbot_python/blob/main/gamepad_utils.py
"""

import enum
import time
import vgamepad as vg

from tclogger import TCLogger
from typing import Union


logger = TCLogger(name="GTAVGamepad", use_prefix=True, use_prefix_ms=True)


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
    CROSS_UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
    CROSS_DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
    CROSS_LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
    CROSS_RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
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


class TriggerPressure:
    """常用扳机压力值映射"""

    RELEASED = 0.0  # 完全松开
    LIGHT = 0.4  # 轻压 (适用于需要精确控制的场景，如半按加速)
    FULL = 1.0  # 完全按下 (适用于射击等场景)


def sleep_ms(ms: int):
    """睡眠指定毫秒数"""
    time.sleep(ms / 1000.0)


class GamepadSimulator:
    """模拟手柄操作"""

    def __init__(self):
        self.pad = None
        self._init_gamepad()

    # ================== 手柄初始化 =================== #

    def _init_gamepad(self):
        """初始化手柄"""
        self._create_gamepad()
        self._reset_gamepad()
        self._wake_gamepad()
        logger.okay("手柄初始化完成")

    def _create_gamepad(self):
        """创建手柄设备"""
        try:
            self.pad = vg.VX360Gamepad()
            logger.okay("手柄设备已创建")
        except Exception as e:
            logger.warn(f"手柄初始化失败: {e}")
            logger.note("请确保已安装 ViGEmBus 驱动，且没有其他程序正在使用")
            logger.note("参考安装文件:")
            logger.file(
                "* https://github.com/shibeta/JNTMbot_python/blob/main/install_vigembus.bat"
            )
            logger.file(
                "* https://github.com/shibeta/JNTMbot_python/blob/main/assets/ViGEmBusSetup_x64.msi"
            )
            raise

    def _reset_gamepad(self):
        """重置手柄状态"""
        if self.pad:
            try:
                logger.mesg("正在重置手柄状态...")
                self.pad.reset()
                self.pad.update()
                logger.mesg("手柄状态已重置")
            except Exception as e:
                logger.warn(f"重置手柄失败: {e}")

    def _wake_gamepad(self):
        """唤醒手柄"""
        self.press_left_joystick(JoystickDirection.HALF_LEFT, duration_ms=10)

    # ================ 手柄生命周期管理 ================= #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理资源"""
        self._reset_gamepad()
        return False

    def __del__(self):
        """析构函数，确保资源释放"""
        self._reset_gamepad()

    # ================ 按钮操作 ================ #
    def hold_button(self, button: AnyButton):
        """按住按钮（不松开）"""
        try:
            self.pad.press_button(button)
            self.pad.update()
        except Exception as e:
            logger.warn(f"按住按钮失败: {e}")

    def release_button(self, button: AnyButton):
        """松开按钮"""
        try:
            self.pad.release_button(button)
            self.pad.update()
        except Exception as e:
            logger.warn(f"松开按钮失败: {e}")

    def press_button(self, button: AnyButton, duration_ms: int = 150):
        """按下按钮，一段时间后松开"""
        try:
            self.hold_button(button)
            sleep_ms(duration_ms)
        except Exception as e:
            logger.warn(f"按下按钮失败: {e}")
        finally:
            self.release_button(button)

    def click_button(self, button: AnyButton, duration_ms: int = 150):
        """快速点击按钮，默认持续时间为150毫秒"""
        self.press_button(button, duration_ms)

    # =============== 左摇杆操作 ================ #
    def _set_left_joystick(self, direction: JoystickDirection):
        """设置左摇杆值（-1.0 ~ 1.0）"""
        try:
            self.pad.left_joystick_float(*direction)
            self.pad.update()
        except Exception as e:
            logger.warn(f"设置左摇杆失败: {e}")

    def hold_left_joystick(self, direction: JoystickDirection):
        """按住左摇杆（-1.0 ~ 1.0）"""
        try:
            self._set_left_joystick(direction)
        except Exception as e:
            logger.warn(f"按住左摇杆失败: {e}")

    def center_left_joystick(self):
        """回中左摇杆"""
        try:
            self._set_left_joystick(JoystickDirection.CENTER)
        except Exception as e:
            logger.warn(f"回中左摇杆失败: {e}")

    def press_left_joystick(self, direction: JoystickDirection, duration_ms: int = 100):
        """按下左摇杆，保持一段时间后回中"""
        try:
            self.hold_left_joystick(direction)
            sleep_ms(duration_ms)
        except Exception as e:
            logger.warn(f"按下左摇杆失败: {e}")
        finally:
            self.center_left_joystick()

    # =============== 右摇杆操作 ================ #
    def _set_right_joystick(self, direction: JoystickDirection):
        """设置右摇杆值（-1.0 ~ 1.0）"""
        try:
            self.pad.right_joystick_float(*direction)
            self.pad.update()
        except Exception as e:
            logger.warn(f"设置右摇杆失败: {e}")

    def hold_right_joystick(self, direction: JoystickDirection):
        """按住右摇杆（-1.0 ~ 1.0）"""
        try:
            self._set_right_joystick(direction)
        except Exception as e:
            logger.warn(f"按住右摇杆失败: {e}")

    def center_right_joystick(self):
        """回中右摇杆"""
        try:
            self._set_right_joystick(JoystickDirection.CENTER)
        except Exception as e:
            logger.warn(f"回中右摇杆失败: {e}")

    def press_right_joystick(
        self, direction: JoystickDirection, duration_ms: int = 100
    ):
        """按下右摇杆，保持一段时间后回中"""
        try:
            self.hold_right_joystick(direction)
            sleep_ms(duration_ms)
        except Exception as e:
            logger.warn(f"按下右摇杆失败: {e}")
        finally:
            self.center_right_joystick()

    # =============== 左扳机操作 ================ #
    def _set_left_trigger(self, pressure: float):
        """设置左扳机 (0.0 ~ 1.0)，效果类似按住"""
        try:
            self.pad.left_trigger_float(pressure)
            self.pad.update()
        except Exception as e:
            logger.warn(f"设置左扳机值失败: {e}")

    def hold_left_trigger(self, pressure: float):
        """按住左扳机 (0.0 ~ 1.0)"""
        try:
            self._set_left_trigger(pressure)
        except Exception as e:
            logger.warn(f"按压左扳机失败: {e}")

    def release_left_trigger(self):
        """松开左扳机"""
        try:
            self._set_left_trigger(TriggerPressure.RELEASED)
        except Exception as e:
            logger.warn(f"松开左扳机失败: {e}")

    def press_left_trigger(self, pressure: float, duration_ms: int = 100):
        """按下左扳机 (0.0 ~ 1.0)，一段时间后松开"""
        try:
            self.hold_left_trigger(pressure)
            sleep_ms(duration_ms)
        except Exception as e:
            logger.warn(f"按下左扳机失败: {e}")
        finally:
            self.release_left_trigger()

    # =============== 右扳机操作 ================ #
    def _set_right_trigger(self, pressure: float):
        """设置右扳机 (0.0 ~ 1.0)，效果类似按住"""
        try:
            self.pad.right_trigger_float(pressure)
            self.pad.update()
        except Exception as e:
            logger.warn(f"设置右扳机值失败: {e}")

    def hold_right_trigger(self, pressure: float):
        """按住右扳机 (0.0 ~ 1.0)"""
        try:
            self._set_right_trigger(pressure)
        except Exception as e:
            logger.warn(f"按压右扳机失败: {e}")

    def release_right_trigger(self):
        """松开右扳机"""
        try:
            self._set_right_trigger(TriggerPressure.RELEASED)
        except Exception as e:
            logger.warn(f"松开右扳机失败: {e}")

    def press_right_trigger(self, pressure: float, duration_ms: int = 100):
        """按下右扳机 (0.0 ~ 1.0)，一段时间后松开"""
        try:
            self.hold_right_trigger(pressure)
            sleep_ms(duration_ms)
        except Exception as e:
            logger.warn(f"按下右扳机失败: {e}")
        finally:
            self.release_right_trigger()


def test_gamepad_simulator():
    """测试手柄功能"""
    with GamepadSimulator() as simulator:
        logger.note("测试：点击 A 键 ...")
        simulator.press_button(Button.A)
        logger.note("测试：按住 B 键 500ms ...")
        simulator.press_left_joystick(JoystickDirection.FULL_UP, duration_ms=500)
        logger.note("测试：按住右扳机（全按）500ms ...")
        simulator.press_right_trigger(TriggerPressure.FULL, duration_ms=500)


if __name__ == "__main__":
    test_gamepad_simulator()

    # python -m gtaz.gamepads
