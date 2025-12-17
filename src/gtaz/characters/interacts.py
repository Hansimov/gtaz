"""GTAV 人物交互模块"""

from tclogger import TCLogger
from time import sleep

from ..gamepads import GamepadSimulator
from ..gamepads import Button, JoystickDirection, TriggerPressure
from ..gamepads import sleep_ms


logger = TCLogger(name="GTAVCharacterInteractor", use_prefix=True, use_prefix_ms=True)


class CharacterInteractor:
    """GTAV 人物交互"""

    def __init__(self, gamepad: GamepadSimulator = None):
        self.gamepad = gamepad or GamepadSimulator()

    # ==================== 常用操作封装 ====================
    def wait_until_ready(self, duration_ms: int = 200):
        """等待以确保操作生效"""
        sleep_ms(duration_ms)

    def press_left_joystick(self, direction: JoystickDirection, duration_ms: int = 100):
        """按下左摇杆"""
        self.gamepad.press_left_joystick(direction, duration_ms)

    def press_right_joystick(
        self, direction: JoystickDirection, duration_ms: int = 100
    ):
        """按下右摇杆"""
        self.gamepad.press_right_joystick(direction, duration_ms)

    def click_button(self, button: Button):
        """点击指定按钮"""
        self.gamepad.click_button(button)

    # ==================== 走动 ====================
    def walk_forward(self, duration_ms: int = 500) -> None:
        """向前走"""
        self.press_left_joystick(JoystickDirection.HALF_UP, duration_ms)

    def walk_backward(self, duration_ms: int = 500) -> None:
        """向后走"""
        self.press_left_joystick(JoystickDirection.HALF_DOWN, duration_ms)

    def walk_left(self, duration_ms: int = 500) -> None:
        """向左走"""
        self.press_left_joystick(JoystickDirection.HALF_LEFT, duration_ms)

    def walk_right(self, duration_ms: int = 500) -> None:
        """向右走"""
        self.press_left_joystick(JoystickDirection.HALF_RIGHT, duration_ms)

    # ==================== 跑动 ====================
    def run_forward(self, duration_ms: int = 500) -> None:
        """向前跑"""
        self.press_left_joystick(JoystickDirection.FULL_UP, duration_ms)

    def run_backward(self, duration_ms: int = 500) -> None:
        """向后跑"""
        self.press_left_joystick(JoystickDirection.FULL_DOWN, duration_ms)

    def run_left(self, duration_ms: int = 500) -> None:
        """向左跑"""
        self.press_left_joystick(JoystickDirection.FULL_LEFT, duration_ms)

    def run_right(self, duration_ms: int = 500) -> None:
        """向右跑"""
        self.press_left_joystick(JoystickDirection.FULL_RIGHT, duration_ms)

    def sprint(self) -> None:
        """冲刺（连按 A 键）"""
        self.click_button(Button.A)

    def stop_moving(self) -> None:
        """停止移动（摇杆回中）"""
        self.gamepad.center_left_joystick()

    # ==================== 视角 ====================
    def look_up(self, duration_ms: int = 300) -> None:
        """视角向上看"""
        self.press_right_joystick(JoystickDirection.FULL_UP, duration_ms)

    def look_down(self, duration_ms: int = 300) -> None:
        """视角向下看"""
        self.press_right_joystick(JoystickDirection.FULL_DOWN, duration_ms)

    def look_left(self, duration_ms: int = 300) -> None:
        """视角向左看"""
        self.press_right_joystick(JoystickDirection.FULL_LEFT, duration_ms)

    def look_right(self, duration_ms: int = 300) -> None:
        """视角向右看"""
        self.press_right_joystick(JoystickDirection.FULL_RIGHT, duration_ms)

    def reset_camera(self) -> None:
        """重置视角（按下右摇杆）"""
        self.click_button(Button.RIGHT_STICK)

    # ==================== 动作 ====================
    def jump(self) -> None:
        """跳跃（X 键）"""
        self.click_button(Button.X)

    def take_cover(self) -> None:
        """进入掩体（RB 键）"""
        self.click_button(Button.RIGHT_SHOULDER)

    def crouch(self) -> None:
        """蹲下（按下左摇杆）"""
        self.click_button(Button.LEFT_STICK)

    def dodge(self) -> None:
        """闪避/翻滚（X 键，瞄准时）"""
        self.click_button(Button.X)

    def melee_attack(self) -> None:
        """近战攻击（B 键）"""
        self.click_button(Button.B)

    # ==================== 武器 ====================
    def aim(self) -> None:
        """瞄准（按住左扳机）"""
        self.gamepad.hold_left_trigger(TriggerPressure.FULL)

    def stop_aiming(self) -> None:
        """停止瞄准（松开左扳机）"""
        self.gamepad.release_left_trigger()

    def shoot(self) -> None:
        """射击（按下右扳机）"""
        self.gamepad.hold_right_trigger(TriggerPressure.FULL)

    def stop_shooting(self) -> None:
        """停止射击（松开右扳机）"""
        self.gamepad.release_right_trigger()

    def aim_and_shoot(self, duration_ms: int = 200) -> None:
        """瞄准并射击"""
        self.aim()
        sleep_ms(50)
        self.gamepad.press_right_trigger(TriggerPressure.FULL, duration_ms)
        self.stop_aiming()

    def reload(self) -> None:
        """装弹（B 键）"""
        self.click_button(Button.B)

    def switch_weapon(self) -> None:
        """切换武器（Y 键）"""
        self.click_button(Button.Y)

    def open_weapon_wheel(self) -> None:
        """打开武器轮盘（按住 LB 键）"""
        self.gamepad.press_button(Button.LEFT_SHOULDER)
        self.wait_until_ready()

    def close_weapon_wheel(self) -> None:
        """关闭武器轮盘（松开 LB 键）"""
        self.gamepad.release_button(Button.LEFT_SHOULDER)
        self.wait_until_ready()

    def throw_grenade(self) -> None:
        """投掷手雷（按 RB 键）"""
        self.click_button(Button.RIGHT_SHOULDER)

    # ==================== 物体交互 ====================
    def enter_vehicle(self) -> None:
        """进入载具（Y 键）"""
        self.click_button(Button.Y)

    def interact(self) -> None:
        """与物体/NPC 交互（Y 键）"""
        self.click_button(Button.Y)

    def pick_up(self) -> None:
        """拾取物品（Y 键）"""
        self.click_button(Button.Y)

    # ==================== 角色切换 ====================
    def switch_character(self) -> None:
        """打开角色切换轮盘（按住下方向键）"""
        self.gamepad.press_button(Button.DPAD_DOWN)

    def close_character_wheel(self) -> None:
        """关闭角色切换轮盘"""
        self.gamepad.release_button(Button.DPAD_DOWN)
        self.wait_until_ready()


class CharacterInteractorTester:
    """人物交互测试"""

    def test(self):
        character = CharacterInteractor()

        logger.note("测试：人物移动 ...")
        character.walk_forward(duration_ms=500)
        sleep(0.5)
        character.look_left(duration_ms=500)
        sleep(0.5)
        character.walk_backward(duration_ms=500)
        sleep(0.5)
        character.look_right(duration_ms=500)
        sleep(0.5)

        logger.okay("测试完成")

    def test_keep_moving(self):
        character = CharacterInteractor()
        logger.note("测试：人物保持移动 ...")
        for i in range(2000):
            character.walk_forward(duration_ms=500)
            sleep(10)
            character.look_left(duration_ms=500)
            sleep(10)
            character.look_right(duration_ms=500)
            sleep(10)
        logger.okay("测试完成")


def test_character_interactor():
    tester = CharacterInteractorTester()
    tester.test()


def test_keep_moving():
    tester = CharacterInteractorTester()
    tester.test_keep_moving()


if __name__ == "__main__":
    # test_character_interactor()
    test_keep_moving()

    # python -m gtaz.characters.interacts
