"""GTAV 控制器 - 高层行为接口"""

import time

from tclogger import TCLogger

from .gamepads import GamepadSimulator, Button, JoystickDirection, TriggerPressure


logger = TCLogger(name="GTAVController", use_prefix=True)


class MenuController:
    """
    GTAV 菜单/界面控制器。

    用于控制游戏菜单、互动菜单、手机等界面操作。
    """

    def __init__(self, gamepad: GamepadSimulator):
        """
        初始化菜单控制器。

        :param gamepad: 手柄模拟器实例
        """
        self.gamepad = gamepad

    def open_pause_menu(self) -> None:
        """打开暂停菜单（START 键）。"""
        self.gamepad.click_button(Button.START)

    def close_menu(self) -> None:
        """关闭当前菜单（B 键）。"""
        self.gamepad.click_button(Button.B)

    def open_interaction_menu(self, hold_duration_ms: int = 1000) -> None:
        """
        打开互动菜单（长按 BACK/SELECT 键）。

        :param hold_duration_ms: 按住持续时间（毫秒）
        """
        self.gamepad.click_button(Button.BACK, hold_duration_ms)

    def confirm(self) -> None:
        """确认选择（A 键）。"""
        self.gamepad.click_button(Button.A)

    def cancel(self) -> None:
        """取消/返回（B 键）。"""
        self.gamepad.click_button(Button.B)

    def navigate_up(self) -> None:
        """菜单向上导航。"""
        self.gamepad.click_button(Button.DPAD_UP)

    def navigate_down(self) -> None:
        """菜单向下导航。"""
        self.gamepad.click_button(Button.DPAD_DOWN)

    def navigate_left(self) -> None:
        """菜单向左导航。"""
        self.gamepad.click_button(Button.DPAD_LEFT)

    def navigate_right(self) -> None:
        """菜单向右导航。"""
        self.gamepad.click_button(Button.DPAD_RIGHT)

    def tab_left(self) -> None:
        """切换到左侧标签页（LB 键）。"""
        self.gamepad.click_button(Button.LEFT_SHOULDER)

    def tab_right(self) -> None:
        """切换到右侧标签页（RB 键）。"""
        self.gamepad.click_button(Button.RIGHT_SHOULDER)

    def open_phone(self) -> None:
        """打开手机（上方向键）。"""
        self.gamepad.click_button(Button.DPAD_UP)

    def close_phone(self) -> None:
        """关闭手机（B 键）。"""
        self.gamepad.click_button(Button.B)


class CharacterController:
    """
    GTAV 人物控制器。

    用于控制步行状态下的角色行为。
    """

    def __init__(self, gamepad: GamepadSimulator):
        """
        初始化人物控制器。

        :param gamepad: 手柄模拟器实例
        """
        self.gamepad = gamepad

    # ==================== 移动 ====================

    def walk_forward(self, duration_ms: int = 500) -> None:
        """
        向前走。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.HALF_UP, duration_ms)

    def walk_backward(self, duration_ms: int = 500) -> None:
        """
        向后走。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.HALF_DOWN, duration_ms)

    def walk_left(self, duration_ms: int = 500) -> None:
        """
        向左走。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.HALF_LEFT, duration_ms)

    def walk_right(self, duration_ms: int = 500) -> None:
        """
        向右走。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.HALF_RIGHT, duration_ms)

    def run_forward(self, duration_ms: int = 500) -> None:
        """
        向前跑。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.FULL_UP, duration_ms)

    def run_backward(self, duration_ms: int = 500) -> None:
        """
        向后跑。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.FULL_DOWN, duration_ms)

    def run_left(self, duration_ms: int = 500) -> None:
        """
        向左跑。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.FULL_LEFT, duration_ms)

    def run_right(self, duration_ms: int = 500) -> None:
        """
        向右跑。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_left_joystick(JoystickDirection.FULL_RIGHT, duration_ms)

    def sprint(self) -> None:
        """冲刺（连按 A 键）。"""
        self.gamepad.click_button(Button.A)

    def stop_moving(self) -> None:
        """停止移动（摇杆回中）。"""
        self.gamepad.return_left_joystick_to_center()

    # ==================== 视角 ====================

    def look_up(self, duration_ms: int = 300) -> None:
        """
        视角向上看。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_right_joystick(JoystickDirection.FULL_UP, duration_ms)

    def look_down(self, duration_ms: int = 300) -> None:
        """
        视角向下看。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_right_joystick(JoystickDirection.FULL_DOWN, duration_ms)

    def look_left(self, duration_ms: int = 300) -> None:
        """
        视角向左看。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_right_joystick(JoystickDirection.FULL_LEFT, duration_ms)

    def look_right(self, duration_ms: int = 300) -> None:
        """
        视角向右看。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.hold_right_joystick(JoystickDirection.FULL_RIGHT, duration_ms)

    def reset_camera(self) -> None:
        """重置视角（按下右摇杆）。"""
        self.gamepad.click_button(Button.RIGHT_STICK)

    # ==================== 动作 ====================

    def jump(self) -> None:
        """跳跃（X 键）。"""
        self.gamepad.click_button(Button.X)

    def take_cover(self) -> None:
        """进入掩体（RB 键）。"""
        self.gamepad.click_button(Button.RIGHT_SHOULDER)

    def crouch(self) -> None:
        """蹲下（按下左摇杆）。"""
        self.gamepad.click_button(Button.LEFT_STICK)

    def dodge(self) -> None:
        """闪避/翻滚（X 键，瞄准时）。"""
        self.gamepad.click_button(Button.X)

    def melee_attack(self) -> None:
        """近战攻击（B 键）。"""
        self.gamepad.click_button(Button.B)

    # ==================== 武器 ====================

    def aim(self) -> None:
        """瞄准（按住左扳机）。"""
        self.gamepad.press_left_trigger(TriggerPressure.full)

    def stop_aiming(self) -> None:
        """停止瞄准（松开左扳机）。"""
        self.gamepad.release_left_trigger()

    def shoot(self) -> None:
        """射击（按下右扳机）。"""
        self.gamepad.press_right_trigger(TriggerPressure.full)

    def stop_shooting(self) -> None:
        """停止射击（松开右扳机）。"""
        self.gamepad.release_right_trigger()

    def aim_and_shoot(self, duration_ms: int = 200) -> None:
        """
        瞄准并射击。

        :param duration_ms: 射击持续时间（毫秒）
        """
        self.aim()
        time.sleep(0.05)
        self.gamepad.hold_right_trigger(TriggerPressure.full, duration_ms)
        self.stop_aiming()

    def reload(self) -> None:
        """装弹（B 键）。"""
        self.gamepad.click_button(Button.B)

    def switch_weapon(self) -> None:
        """切换武器（Y 键）。"""
        self.gamepad.click_button(Button.Y)

    def open_weapon_wheel(self) -> None:
        """打开武器轮盘（按住 LB 键）。"""
        self.gamepad.press_button(Button.LEFT_SHOULDER)

    def close_weapon_wheel(self) -> None:
        """关闭武器轮盘（松开 LB 键）。"""
        self.gamepad.release_button(Button.LEFT_SHOULDER)

    def throw_grenade(self) -> None:
        """投掷手雷（按 RB 键）。"""
        self.gamepad.click_button(Button.RIGHT_SHOULDER)

    # ==================== 交互 ====================

    def enter_vehicle(self) -> None:
        """进入载具（Y 键）。"""
        self.gamepad.click_button(Button.Y)

    def interact(self) -> None:
        """与物体/NPC 交互（Y 键）。"""
        self.gamepad.click_button(Button.Y)

    def pick_up(self) -> None:
        """拾取物品（Y 键）。"""
        self.gamepad.click_button(Button.Y)

    # ==================== 角色切换 ====================

    def switch_character(self) -> None:
        """打开角色切换轮盘（按住下方向键）。"""
        self.gamepad.press_button(Button.DPAD_DOWN)

    def close_character_wheel(self) -> None:
        """关闭角色切换轮盘。"""
        self.gamepad.release_button(Button.DPAD_DOWN)

    def quick_save(self) -> None:
        """快速存档（手机 -> 快速存档）。"""
        # 打开手机
        self.gamepad.click_button(Button.DPAD_UP)
        time.sleep(0.5)
        # 这里只是打开手机，具体存档操作需要进一步导航


class VehicleController:
    """
    GTAV 载具控制器。

    用于控制驾驶状态下的载具行为。
    预留接口，待后续实现。
    """

    def __init__(self, gamepad: GamepadSimulator):
        """
        初始化载具控制器。

        :param gamepad: 手柄模拟器实例
        """
        self.gamepad = gamepad

    # ==================== 基础驾驶 ====================

    def accelerate(self, pressure: float = 1.0) -> None:
        """
        加速（右扳机）。

        :param pressure: 油门压力，0.0 ~ 1.0
        """
        self.gamepad.press_right_trigger(pressure)

    def release_accelerator(self) -> None:
        """松开油门。"""
        self.gamepad.release_right_trigger()

    def brake(self, pressure: float = 1.0) -> None:
        """
        刹车/倒车（左扳机）。

        :param pressure: 刹车压力，0.0 ~ 1.0
        """
        self.gamepad.press_left_trigger(pressure)

    def release_brake(self) -> None:
        """松开刹车。"""
        self.gamepad.release_left_trigger()

    def steer_left(self, intensity: float = 1.0) -> None:
        """
        向左转向。

        :param intensity: 转向强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((-intensity, 0.0))

    def steer_right(self, intensity: float = 1.0) -> None:
        """
        向右转向。

        :param intensity: 转向强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((intensity, 0.0))

    def steer_straight(self) -> None:
        """方向盘回正。"""
        self.gamepad.return_left_joystick_to_center()

    def handbrake(self) -> None:
        """手刹（RB 键）。"""
        self.gamepad.press_button(Button.RIGHT_SHOULDER)

    def release_handbrake(self) -> None:
        """松开手刹。"""
        self.gamepad.release_button(Button.RIGHT_SHOULDER)

    def handbrake_turn(self, duration_ms: int = 500) -> None:
        """
        手刹漂移。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.click_button(Button.RIGHT_SHOULDER, duration_ms)

    # ==================== 视角 ====================

    def look_behind(self) -> None:
        """向后看（按下右摇杆）。"""
        self.gamepad.click_button(Button.RIGHT_STICK)

    def change_camera(self) -> None:
        """切换视角（按下 SELECT/BACK 键）。"""
        self.gamepad.click_button(Button.BACK)

    # ==================== 载具功能 ====================

    def honk(self, duration_ms: int = 300) -> None:
        """
        鸣笛（按下左摇杆）。

        :param duration_ms: 持续时间（毫秒）
        """
        self.gamepad.click_button(Button.LEFT_STICK, duration_ms)

    def toggle_headlights(self) -> None:
        """切换车灯（按下右方向键）。"""
        self.gamepad.click_button(Button.DPAD_RIGHT)

    def toggle_radio(self) -> None:
        """切换电台（按下左方向键）。"""
        self.gamepad.click_button(Button.DPAD_LEFT)

    def exit_vehicle(self) -> None:
        """下车（Y 键）。"""
        self.gamepad.click_button(Button.Y)

    def cinematic_camera(self) -> None:
        """电影视角（按住 SELECT/BACK 键）。"""
        self.gamepad.click_button(Button.BACK, 500)

    # ==================== 载具战斗 ====================

    def drive_by_aim(self) -> None:
        """车内瞄准（按住 LB 键）。"""
        self.gamepad.press_button(Button.LEFT_SHOULDER)

    def stop_drive_by_aim(self) -> None:
        """停止车内瞄准。"""
        self.gamepad.release_button(Button.LEFT_SHOULDER)

    def drive_by_shoot(self) -> None:
        """车内射击（按下 A 键，瞄准时）。"""
        self.gamepad.click_button(Button.A)

    # ==================== 特殊载具 ====================

    def aircraft_pitch_up(self, intensity: float = 1.0) -> None:
        """
        飞行器抬头。

        :param intensity: 强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((0.0, -intensity))

    def aircraft_pitch_down(self, intensity: float = 1.0) -> None:
        """
        飞行器俯冲。

        :param intensity: 强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((0.0, intensity))

    def aircraft_roll_left(self, intensity: float = 1.0) -> None:
        """
        飞行器左滚。

        :param intensity: 强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((-intensity, 0.0))

    def aircraft_roll_right(self, intensity: float = 1.0) -> None:
        """
        飞行器右滚。

        :param intensity: 强度，0.0 ~ 1.0
        """
        self.gamepad.move_left_joystick((intensity, 0.0))

    def aircraft_yaw_left(self) -> None:
        """飞行器左偏航（LB 键）。"""
        self.gamepad.press_button(Button.LEFT_SHOULDER)

    def aircraft_yaw_right(self) -> None:
        """飞行器右偏航（RB 键）。"""
        self.gamepad.press_button(Button.RIGHT_SHOULDER)

    def aircraft_landing_gear(self) -> None:
        """收放起落架（A 键）。"""
        self.gamepad.click_button(Button.A)

    def helicopter_ascend(self) -> None:
        """直升机上升（右扳机）。"""
        self.gamepad.press_right_trigger(TriggerPressure.full)

    def helicopter_descend(self) -> None:
        """直升机下降（左扳机）。"""
        self.gamepad.press_left_trigger(TriggerPressure.full)

    def boat_boost(self) -> None:
        """船只加速（A 键）。"""
        self.gamepad.click_button(Button.A)


def test_controllers():
    """测试控制器。"""
    gamepad = GamepadSimulator()

    menu = MenuController(gamepad)
    character = CharacterController(gamepad)
    vehicle = VehicleController(gamepad)

    logger.note("测试：打开暂停菜单 ...")
    menu.open_pause_menu()
    time.sleep(3)
    menu.close_menu()
    time.sleep(0.5)

    logger.note("测试：打开手机 ...")
    menu.open_phone()
    time.sleep(3)
    menu.close_phone()
    time.sleep(0.5)

    logger.note("测试：人物移动 ...")
    character.walk_forward(duration_ms=500)
    time.sleep(0.5)
    character.look_left(duration_ms=500)
    time.sleep(0.5)
    character.walk_backward(duration_ms=500)
    time.sleep(0.5)
    character.look_right(duration_ms=500)
    time.sleep(0.5)

    logger.okay("测试完成")


def test_keep_moving():
    gamepad = GamepadSimulator()
    character = CharacterController(gamepad)
    logger.note("测试：人物保持移动 ...")
    for i in range(2000):
        character.walk_forward(duration_ms=500)
        time.sleep(10)
        character.look_left(duration_ms=500)
        time.sleep(10)
        character.walk_backward(duration_ms=500)
        time.sleep(10)
        character.look_right(duration_ms=500)
        time.sleep(10)
    logger.okay("测试完成")


if __name__ == "__main__":
    # test_controllers()
    test_keep_moving()

    # python -m gtaz.controls
