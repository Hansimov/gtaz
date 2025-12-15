from pathlib import Path
from tclogger import strf_path, logstr

MENU_IMGS_DIR = Path(__file__).parent / "imgs"

LV1_INFOS = [
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
