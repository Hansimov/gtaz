"""GTAV 键盘动作检测"""

import ctypes
import time
from dataclasses import dataclass, field
from typing import Optional

from tclogger import TCLogger, get_now


logger = TCLogger(name="KeyboardActionDetector", use_prefix=True, use_prefix_ms=True)


# 虚拟键码映射（常用按键）
VK_CODES = {
    # 字母键
    0x41: "A",
    0x42: "B",
    0x43: "C",
    0x44: "D",
    0x45: "E",
    0x46: "F",
    0x47: "G",
    0x48: "H",
    0x49: "I",
    0x4A: "J",
    0x4B: "K",
    0x4C: "L",
    0x4D: "M",
    0x4E: "N",
    0x4F: "O",
    0x50: "P",
    0x51: "Q",
    0x52: "R",
    0x53: "S",
    0x54: "T",
    0x55: "U",
    0x56: "V",
    0x57: "W",
    0x58: "X",
    0x59: "Y",
    0x5A: "Z",
    # 数字键
    0x30: "0",
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
    0x35: "5",
    0x36: "6",
    0x37: "7",
    0x38: "8",
    0x39: "9",
    # 功能键
    0x70: "F1",
    0x71: "F2",
    0x72: "F3",
    0x73: "F4",
    0x74: "F5",
    0x75: "F6",
    0x76: "F7",
    0x77: "F8",
    0x78: "F9",
    0x79: "F10",
    0x7A: "F11",
    0x7B: "F12",
    # 控制键
    0x08: "Backspace",
    0x09: "Tab",
    0x0D: "Enter",
    0x1B: "Escape",
    0x20: "Space",
    0x21: "PageUp",
    0x22: "PageDown",
    0x23: "End",
    0x24: "Home",
    0x25: "Left",
    0x26: "Up",
    0x27: "Right",
    0x28: "Down",
    0x2D: "Insert",
    0x2E: "Delete",
    # 修饰键
    0x10: "Shift",
    0x11: "Ctrl",
    0x12: "Alt",
    0xA0: "LShift",
    0xA1: "RShift",
    0xA2: "LCtrl",
    0xA3: "RCtrl",
    0xA4: "LAlt",
    0xA5: "RAlt",
    # 数字小键盘
    0x60: "Numpad0",
    0x61: "Numpad1",
    0x62: "Numpad2",
    0x63: "Numpad3",
    0x64: "Numpad4",
    0x65: "Numpad5",
    0x66: "Numpad6",
    0x67: "Numpad7",
    0x68: "Numpad8",
    0x69: "Numpad9",
    0x6A: "Multiply",
    0x6B: "Add",
    0x6D: "Subtract",
    0x6E: "Decimal",
    0x6F: "Divide",
    # 其他
    0xBE: ".",
    0xBC: ",",
    0xBD: "-",
    0xBB: "=",
    0xBA: ";",
    0xDE: "'",
    0xC0: "`",
    0xDB: "[",
    0xDD: "]",
    0xDC: "\\",
    0xBF: "/",
    0x14: "CapsLock",
    0x90: "NumLock",
    0x91: "ScrollLock",
}

# 需要监控的按键范围（0-255）
MONITORED_KEY_RANGE = range(256)

# GTAV 常用游戏按键（可根据需求调整）
GTAV_GAME_KEYS = [
    0x57,  # W - 前进
    0x41,  # A - 左移
    0x53,  # S - 后退
    0x44,  # D - 右移
    0x20,  # Space - 跳跃/手刹
    0x10,  # Shift - 奔跑/加速
    0x11,  # Ctrl - 蹲下
    0x45,  # E - 进入载具/互动
    0x46,  # F - 进入载具（备用）
    0x51,  # Q - 掩护
    0x52,  # R - 换弹
    0x47,  # G - 投掷武器
    0x54,  # T - 手机
    0x4D,  # M - 地图
    0x09,  # Tab - 选择武器
    0x1B,  # Escape - 菜单
    0x25,  # Left - 方向键左
    0x26,  # Up - 方向键上
    0x27,  # Right - 方向键右
    0x28,  # Down - 方向键下
]


@dataclass
class KeyState:
    """单个按键的状态信息。"""

    key_code: int
    key_name: str
    is_pressed: bool = False
    press_time: Optional[float] = None
    release_time: Optional[float] = None

    def copy(self) -> "KeyState":
        """创建当前状态的副本。"""
        return KeyState(
            key_code=self.key_code,
            key_name=self.key_name,
            is_pressed=self.is_pressed,
            press_time=self.press_time,
            release_time=self.release_time,
        )

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "key_code": self.key_code,
            "key_name": self.key_name,
            "is_pressed": self.is_pressed,
            "press_time": self.press_time,
            "release_time": self.release_time,
        }


@dataclass
class KeyboardActionInfo:
    """键盘动作信息。"""

    timestamp: float
    datetime_str: str
    pressed_keys: list[str] = field(default_factory=list)
    key_states: dict[str, KeyState] = field(default_factory=dict)
    has_action: bool = False

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "has_action": self.has_action,
            "pressed_keys": self.pressed_keys,
            "key_states": {k: v.to_dict() for k, v in self.key_states.items()},
        }


class KeyboardActionDetector:
    """
    键盘动作检测器。

    检测当前窗口是否有任何键盘输入，包括按下、抬起、正在按住等状态。
    使用 Windows GetAsyncKeyState API 来检测全局键盘状态。
    """

    def __init__(
        self,
        monitored_keys: Optional[list[int]] = None,
        game_keys_only: bool = False,
    ):
        """
        初始化键盘动作检测器。

        :param monitored_keys: 要监控的按键列表（虚拟键码），默认监控所有按键
        :param game_keys_only: 是否只监控 GTAV 游戏常用按键
        """
        if game_keys_only:
            self.monitored_keys = GTAV_GAME_KEYS
        elif monitored_keys is not None:
            self.monitored_keys = monitored_keys
        else:
            self.monitored_keys = list(MONITORED_KEY_RANGE)

        # 加载 Windows API
        self.user32 = ctypes.windll.user32

        # 按键状态缓存
        self._key_states: dict[int, KeyState] = {}
        self._previous_pressed: set[int] = set()

    def _get_key_name(self, key_code: int) -> str:
        """
        获取按键名称。

        :param key_code: 虚拟键码
        :return: 按键名称
        """
        return VK_CODES.get(key_code, f"VK_{key_code:02X}")

    def _is_key_pressed(self, key_code: int) -> bool:
        """
        检查指定按键是否被按下。

        使用 GetAsyncKeyState 检测按键状态。
        返回值的最高位（0x8000）表示按键当前是否被按下。

        :param key_code: 虚拟键码
        :return: 按键是否被按下
        """
        state = self.user32.GetAsyncKeyState(key_code)
        return bool(state & 0x8000)

    def get_pressed_keys(self) -> list[int]:
        """
        获取当前所有被按下的按键。

        :return: 被按下的按键列表（虚拟键码）
        """
        pressed = []
        for key_code in self.monitored_keys:
            if self._is_key_pressed(key_code):
                pressed.append(key_code)
        return pressed

    def has_any_key_pressed(self) -> bool:
        """
        检查是否有任何按键被按下。

        :return: 是否有按键被按下
        """
        for key_code in self.monitored_keys:
            if self._is_key_pressed(key_code):
                return True
        return False

    def detect(self) -> KeyboardActionInfo:
        """
        检测当前键盘动作状态。

        返回包含所有按键状态的信息对象。

        :return: 键盘动作信息
        """
        now = time.time()
        now_dt = get_now()
        datetime_str = (
            now_dt.strftime("%Y-%m-%d %H-%M-%S") + f".{now_dt.microsecond // 1000:03d}"
        )

        current_pressed: set[int] = set()
        pressed_keys: list[str] = []
        key_states: dict[str, KeyState] = {}

        for key_code in self.monitored_keys:
            is_pressed = self._is_key_pressed(key_code)
            key_name = self._get_key_name(key_code)

            if is_pressed:
                current_pressed.add(key_code)
                pressed_keys.append(key_name)

            # 获取或创建按键状态
            if key_code in self._key_states:
                state = self._key_states[key_code]
            else:
                state = KeyState(key_code=key_code, key_name=key_name)
                self._key_states[key_code] = state

            # 更新按键状态
            was_pressed = key_code in self._previous_pressed

            if is_pressed and not was_pressed:
                # 按键刚被按下
                state.is_pressed = True
                state.press_time = now
                state.release_time = None
            elif not is_pressed and was_pressed:
                # 按键刚被释放
                state.is_pressed = False
                state.release_time = now
            elif is_pressed:
                # 按键正在被按住
                state.is_pressed = True

            if is_pressed:
                key_states[key_name] = state.copy()

        # 更新上一次按下状态
        self._previous_pressed = current_pressed

        # 判断是否有动作
        has_action = len(current_pressed) > 0

        return KeyboardActionInfo(
            timestamp=now,
            datetime_str=datetime_str,
            pressed_keys=pressed_keys,
            key_states=key_states,
            has_action=has_action,
        )

    def reset(self):
        """重置所有按键状态。"""
        self._key_states.clear()
        self._previous_pressed.clear()

    def __repr__(self) -> str:
        return (
            f"KeyboardActionDetector("
            f"monitored_keys={len(self.monitored_keys)} keys)"
        )


def test_keyboard_action_detector():
    """测试键盘动作检测器。"""
    detector = KeyboardActionDetector(game_keys_only=True)
    logger.note(f"检测器信息: {detector}")
    logger.note("开始检测键盘动作，按 Ctrl+C 停止...")

    try:
        while True:
            action_info = detector.detect()
            if action_info.has_action:
                logger.okay(f"检测到按键: {action_info.pressed_keys}")
            time.sleep(0.05)  # 50ms 检测间隔
    except KeyboardInterrupt:
        logger.note("检测已停止")


if __name__ == "__main__":
    test_keyboard_action_detector()

    # python -m gtaz.keyboard_actions
