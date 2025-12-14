"""GTAV 键盘动作检测"""

import ctypes
import time
from dataclasses import dataclass, field
from typing import Literal

from tclogger import TCLogger, get_now


logger = TCLogger(name="KeyboardActionDetector", use_prefix=True, use_prefix_ms=True)

KEY_UP = "up"  # 边沿触发：仅在按键刚释放时触发（检测一次）
KEY_DOWN = "down"  # 边沿触发：仅在按键刚按下时触发（检测一次）
KEY_HOLD = "hold"  # 电平触发：按键按下期间持续触发

TriggerType = Literal["up", "down", "hold"]

# 虚拟键码到按键名的映射（主映射）
KEY_CODE_TO_NAME = {
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

# 按键名到虚拟键码的反向映射（从 KEY_CODE_TO_NAME 构建）
KEY_NAME_TO_CODE = {name: code for code, name in KEY_CODE_TO_NAME.items()}

# 所有支持的按键列表
ALL_KEYS = list(KEY_NAME_TO_CODE.keys())


def key_name_to_code(key: str) -> int:
    """将按键名转换为虚拟键码（大小写不敏感）。

    :param key: 按键名
    :return: 虚拟键码，如果不存在返回 0
    """
    # 大小写不敏感查找
    for name, code in KEY_NAME_TO_CODE.items():
        if name.upper() == key.upper():
            return code
    return 0


def key_code_to_name(code: int) -> str:
    """将虚拟键码转换为按键名。

    :param code: 虚拟键码
    :return: 按键名，如果不存在返回 'Unknown'
    """
    return KEY_CODE_TO_NAME.get(code, f"Unknown_{code:02X}")


def normalize_keys(keys: list[str]) -> list[str]:
    """标准化用户输入的按键列表。

    支持：
    - 常见键名：W/A/S/D、Shift/Ctrl/Alt、F1、Escape、Space...
    - 数字键："1".."9"
    """
    normalized: list[str] = []
    for raw in keys:
        key = raw.strip().replace(" ", "")
        if not key:
            continue

        # 查找匹配的键（大小写不敏感）
        key_upper = key.upper()
        matched_key = None
        for k in KEY_NAME_TO_CODE.keys():
            if k.upper() == key_upper:
                matched_key = k
                break

        if matched_key:
            normalized.append(matched_key)
        else:
            raise ValueError(f"无法识别的按键名称: {raw}")

    # 去重，保持顺序
    seen = set()
    deduped: list[str] = []
    for key in normalized:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


# GTAV 常用游戏按键（可根据需求调整）
GTAV_GAME_KEYS = [
    "W",  # 前进
    "A",  # 左移
    "S",  # 后退
    "D",  # 右移
    "Space",  # 跳跃/手刹
    "Shift",  # 奔跑/加速
    "Ctrl",  # 蹲下
    "E",  # 进入载具/互动
    "F",  # 进入载具（备用）
    "Q",  # 掩护
    "R",  # 换弹
    "G",  # 投掷武器
    "T",  # 手机
    "M",  # 地图
    "Tab",  # 选择武器
    "Escape",  # 菜单
    "Left",  # 方向键左
    "Up",  # 方向键上
    "Right",  # 方向键右
    "Down",  # 方向键下
]


@dataclass
class KeyState:
    """单个按键的状态信息。"""

    key: str
    is_pressed: bool = False
    press_time: float = None
    release_time: float = None

    def copy(self) -> "KeyState":
        """创建当前状态的副本。"""
        return KeyState(
            key=self.key,
            is_pressed=self.is_pressed,
            press_time=self.press_time,
            release_time=self.release_time,
        )

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "key": self.key,
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
        monitor_keys: list[str] = None,
        game_keys_only: bool = False,
        trigger_type: TriggerType = None,
        exclude_keys: list[str] = None,
    ):
        """
        初始化键盘动作检测器。

        :param monitor_keys: 要监控的按键列表（键名），默认监控所有按键
        :param game_keys_only: 是否只监控 GTAV 游戏常用按键
        :param trigger_type: 按键触发类型（KEY_DOWN=刚按下/KEY_UP=刚释放/KEY_HOLD=按住）
        :param exclude_keys: 要排除的按键列表（键名），这些按键不会被检测和记录
        """
        if monitor_keys:
            self.monitor_keys = normalize_keys(monitor_keys)
        elif game_keys_only:
            self.monitor_keys = GTAV_GAME_KEYS
        else:
            self.monitor_keys = ALL_KEYS

        if exclude_keys:
            exclude_set = set(normalize_keys(exclude_keys))
            self.monitor_keys = [
                key for key in self.monitor_keys if key not in exclude_set
            ]

        # 设置触发类型，如果未指定则默认为 KEY_DOWN
        self.trigger_type = trigger_type or KEY_DOWN

        # 加载 Windows API
        self.user32 = ctypes.windll.user32

        # 按键状态缓存
        self._key_states: dict[str, KeyState] = {}
        self._previous_pressed: set[str] = set()

        # 初始化时清除所有按键的历史状态
        # 通过调用一次 GetAsyncKeyState 来清除 toggle 位
        for key in self.monitor_keys:
            key_code = key_name_to_code(key)
            if key_code != 0:
                self.user32.GetAsyncKeyState(key_code)

    def _is_key_pressed(self, key: str) -> bool:
        """
        检查指定按键是否被按下（仅检查当前状态）。

        使用 GetAsyncKeyState 检测按键状态。
        返回值的最高位（0x8000）表示按键当前是否被按下。

        :param key: 按键名
        :return: 按键是否被按下
        """
        key_code = key_name_to_code(key)
        if key_code == 0:
            return False
        state = self.user32.GetAsyncKeyState(key_code)
        return bool(state & 0x8000)

    def get_pressed_keys(self) -> list[str]:
        """
        获取当前所有被按下的按键。

        :return: 被按下的按键列表
        """
        pressed = []
        for key in self.monitor_keys:
            if self._is_key_pressed(key):
                pressed.append(key)
        return pressed

    def has_any_key_pressed(self) -> bool:
        """
        检查是否有任何按键被按下。

        :return: 是否有按键被按下
        """
        for key in self.monitor_keys:
            if self._is_key_pressed(key):
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

        current_pressed: set[str] = set()
        pressed_keys: list[str] = []
        key_states: dict[str, KeyState] = {}

        for key in self.monitor_keys:
            is_pressed = self._is_key_pressed(key)

            if is_pressed:
                current_pressed.add(key)
                pressed_keys.append(key)

            # 获取或创建按键状态
            if key in self._key_states:
                state = self._key_states[key]
            else:
                state = KeyState(key=key)
                self._key_states[key] = state

            # 更新按键状态
            was_pressed = key in self._previous_pressed

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
                key_states[key] = state.copy()

        # 根据 trigger_type 判断是否有动作（必须在更新 _previous_pressed 之前判断）
        has_action = self._determine_has_action(current_pressed)

        # 更新上一次按下状态
        self._previous_pressed = current_pressed

        return KeyboardActionInfo(
            timestamp=now,
            datetime_str=datetime_str,
            pressed_keys=pressed_keys,
            key_states=key_states,
            has_action=has_action,
        )

    def _determine_has_action(self, current_pressed: set[str]) -> bool:
        """
        根据 trigger_type 判断是否有动作。

        :param current_pressed: 当前按下的按键集合
        :return: 是否有动作
        """
        trigger_type = self.trigger_type.lower()

        if trigger_type == KEY_HOLD:
            # 电平触发：只要有按键按下就触发
            return len(current_pressed) > 0
        elif trigger_type == KEY_DOWN:
            # 边沿触发（按下）：只有在按键刚按下时触发
            for key in current_pressed:
                if key not in self._previous_pressed:
                    return True
            return False
        elif trigger_type == KEY_UP:
            # 边沿触发（释放）：只有在按键刚释放时触发
            for key in self._previous_pressed:
                if key not in current_pressed:
                    return True
            return False

        return False

    def reset(self):
        """重置所有按键状态。"""
        self._key_states.clear()
        self._previous_pressed.clear()

    def __repr__(self) -> str:
        return (
            f"KeyboardActionDetector("
            f"monitor_keys={len(self.monitor_keys)} keys, "
            f"trigger_type={self.trigger_type.value})"
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
