from pathlib import Path
from tclogger import strf_path, logstr

MENU_IMGS_DIR = Path(__file__).parent / "imgs"

MENU_HEADER_INFOS = [
    {"name": "地图", "img": "lv1_地图.jpg", "level": 1, "index": 0},
    {"name": "在线", "img": "lv1_在线.jpg", "level": 1, "index": 1},
    {"name": "职业", "img": "lv1_职业.jpg", "level": 1, "index": 2},
    {"name": "好友", "img": "lv1_好友.jpg", "level": 1, "index": 3},
    {"name": "信息", "img": "lv1_信息.jpg", "level": 1, "index": 4},
    {"name": "商店", "img": "lv1_商店.jpg", "level": 1, "index": 5},
    {"name": "设置", "img": "lv1_设置.jpg", "level": 1, "index": 6},
    {"name": "统计", "img": "lv1_统计.jpg", "level": 1, "index": 7},
    {"name": "相册", "img": "lv1_相册.jpg", "level": 1, "index": 8},
]

MENU_FOCUS_INFOS = [
    {"name": "地图", "img": "focus_地图.jpg", "level": 1, "index": 0},
    {"name": "在线", "img": "focus_在线.jpg", "level": 1, "index": 1},
    {"name": "职业", "img": "focus_职业.jpg", "level": 1, "index": 2},
    {"name": "好友", "img": "focus_好友.jpg", "level": 1, "index": 3},
    {"name": "信息", "img": "focus_信息.jpg", "level": 1, "index": 4},
    {"name": "商店", "img": "focus_商店.jpg", "level": 1, "index": 5},
    {"name": "设置", "img": "focus_设置.jpg", "level": 1, "index": 6},
    {"name": "统计", "img": "focus_统计.jpg", "level": 1, "index": 7},
    {"name": "相册", "img": "focus_相册.jpg", "level": 1, "index": 8},
]

RESOLUTIONS = [
    (1024, 768),
    (1152, 864),
    (1280, 720),
    (1280, 768),
    (1280, 800),
    (1280, 960),
    (1280, 1024),  # 无边窗口化
    (1360, 768),
    (1366, 768),
    (1440, 900),
    (1440, 1080),  # 无边窗口化
    (1600, 900),
    (1600, 1024),  # 无边窗口化
    (1680, 1050),  # 无边窗口化
    (1904, 1001),
    (1920, 1080),
]


def find_latest_jpg() -> str:
    menus_path = Path(__file__).parents[1] / "cache" / "menus"
    jpgs = list(menus_path.glob("**/*.jpg"))
    sorted_jpgs = sorted(jpgs, key=lambda p: p.stat().st_mtime, reverse=True)
    latest_jpg = sorted_jpgs[0]
    return strf_path(latest_jpg)


def key_note(s) -> str:
    """为键添加消息样式。"""
    return logstr.note(s)


def val_mesg(s) -> str:
    """为值添加消息样式"""
    return logstr.mesg(s)
