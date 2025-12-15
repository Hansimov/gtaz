from pathlib import Path
from tclogger import strf_path, logstr

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


MENU_IMGS_DIR = Path(__file__).parent / "imgs"

# 标题栏
MENU_HEADER_INFOS = [
    {"name": "地图", "img": "header_地图.jpg", "level": 1, "index": 0},
    {"name": "在线", "img": "header_在线.jpg", "level": 1, "index": 1},
    {"name": "职业", "img": "header_职业.jpg", "level": 1, "index": 2},
    {"name": "好友", "img": "header_好友.jpg", "level": 1, "index": 3},
    {"name": "信息", "img": "header_信息.jpg", "level": 1, "index": 4},
    {"name": "商店", "img": "header_商店.jpg", "level": 1, "index": 5},
    {"name": "设置", "img": "header_设置.jpg", "level": 1, "index": 6},
    {"name": "统计", "img": "header_统计.jpg", "level": 1, "index": 7},
    {"name": "相册", "img": "header_相册.jpg", "level": 1, "index": 8},
]

# 标题焦点
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

# 列表最大尺寸
MAX_LIST_SIZE = (330, 485)
# 列表最大可见条目数
MAX_VISIBLE_ITEMS = 16

# (1024x768 分辨率) 列表尺寸拟合公式:
# height = visible * 29 + 25

# 一级菜单列表
MENU_LIST_INFOS = [
    {"name": "在线", "img": "list_在线.jpg", "level": 1, "total": 18},
    {"name": "在线", "img": "list_在线_1.jpg", "level": 1, "total": 18},
    {"name": "在线", "img": "list_在线_2.jpg", "level": 1, "total": 18},
    {"name": "信息", "img": "list_信息.jpg", "level": 1, "total": 4},
    {"name": "设置", "img": "list_设置.jpg", "level": 1, "total": 5},
    {"name": "统计", "img": "list_统计.jpg", "level": 1, "total": 10},
]

# 二级菜单列表: "在线"
LIST_在线_INFOS = [
    {"name": "在线_差事", "img": "list_在线_差事.jpg", "level": 2, "total": 3},
    {"name": "在线_游玩清单", "level": 2, "total": 6},
    {"name": "在线_寻找新战局", "level": 2, "total": 5},
]

# 条目: ["在线"]
ITEM_在线_INFOS = [
    {"name": "差事", "img": "item_在线_差事.jpg", "level": 2, "index": 0},
    {"name": "加入好友", "img": "item_在线_加入好友.jpg", "level": 2, "index": 1},
    {
        "name": "加入帮会成员",
        "img": "item_在线_加入帮会成员.jpg",
        "level": 2,
        "index": 2,
    },
    {"name": "游玩清单", "img": "item_在线_游玩清单.jpg", "level": 2, "index": 3},
    {"name": "玩家", "img": "item_在线_玩家.jpg", "level": 2, "index": 4},
    {"name": "帮会", "img": "item_在线_帮会.jpg", "level": 2, "index": 5},
    {
        "name": "Rockstar制作器",
        "img": "item_在线_Rockstar制作器.jpg",
        "level": 2,
        "index": 6,
    },
    {"name": "管理角色", "img": "item_在线_管理角色.jpg", "level": 2, "index": 7},
    {"name": "迁移档案", "img": "item_在线_迁移档案.jpg", "level": 2, "index": 8},
    {"name": "GTA加会员", "img": "item_在线_GTA加会员.jpg", "level": 2, "index": 9},
    {
        "name": "购买鲨鱼现金卡",
        "img": "item_在线_购买鲨鱼现金卡.jpg",
        "level": 2,
        "index": 10,
    },
    {"name": "安全与提示", "img": "item_在线_安全与提示.jpg", "level": 2, "index": 11},
    {"name": "选项", "img": "item_在线_选项.jpg", "level": 2, "index": 12},
    {"name": "寻找新战局", "img": "item_在线_寻找新战局.jpg", "level": 2, "index": 13},
    {
        "name": "制作人员名单和法律声明",
        "img": "item_在线_制作人员名单和法律声明.jpg",
        "level": 2,
        "index": 14,
    },
    {
        "name": "退至故事模式",
        "img": "item_在线_退至故事模式.jpg",
        "level": 2,
        "index": 15,
    },
    {"name": "退至主菜单", "img": "item_在线_退至主菜单.jpg", "level": 2, "index": 16},
    {"name": "退出游戏", "img": "item_在线_退出游戏.jpg", "level": 2, "index": 17},
]

# 条目: ["在线", "差事"]
ITEM_在线_差事_INFOS = [
    {"name": "快速加入", "img": "item_在线_差事_快速加入.jpg", "level": 3, "index": 0},
    {"name": "进行差事", "img": "item_在线_差事_进行差事.jpg", "level": 3, "index": 1},
    {"name": "举报差事", "img": "item_在线_差事_举报差事.jpg", "level": 3, "index": 2},
]

# 条目: ["在线", "寻找新战局"]
ITEM_在线_寻找新战局_INFOS = [
    {
        "name": "公开战局",
        "img": "item_在线_寻找新战局_公开战局.jpg",
        "level": 3,
        "index": 0,
    },
    {
        "name": "仅限邀请的战局",
        "img": "item_在线_寻找新战局_仅限邀请的战局.jpg",
        "level": 3,
        "index": 1,
    },
    {
        "name": "帮会战局",
        "img": "item_在线_寻找新战局_帮会战局.jpg",
        "level": 3,
        "index": 2,
    },
    {
        "name": "非公开帮会战局",
        "img": "item_在线_寻找新战局_非公开帮会战局.jpg",
        "level": 3,
        "index": 3,
    },
    {
        "name": "非公开好友战局",
        "img": "item_在线_寻找新战局_非公开好友战局.jpg",
        "level": 3,
        "index": 4,
    },
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
