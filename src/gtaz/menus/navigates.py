"""GTAV 菜单导航模块"""

from dataclasses import dataclass
from tclogger import TCLogger
from typing import Union

from .locates import MatchResult, MergedMatchResult
from .commons import MENU_FOCUS_INFOS, MENU_PARENT_TO_ITEM_INFOS
from .commons import is_names_start_with
from .interacts import MenuInteractor
from .locates import MenuLocatorRunner, is_score_too_low, is_score_high
from ..screens import ScreenCapturer

logger = TCLogger(name="GTAVMenuNavigator", use_prefix=True, use_prefix_ms=True)


class LocateNamesConverter:
    @staticmethod
    def _is_list_belongs_to_focus(r: MergedMatchResult) -> bool:
        """判断 focus.names 是否是 list.names 的前缀"""
        return is_names_start_with(tuple(r.list.names), tuple(r.focus.names))

    @staticmethod
    def _is_item_belongs_to_list(r: MergedMatchResult) -> bool:
        """判断 list.names 是否是 item.names 的前缀"""
        return is_names_start_with(tuple(r.item.names), tuple(r.list.names))

    @staticmethod
    def _is_item_belongs_to_focus(r: MergedMatchResult) -> bool:
        """判断focus.names 是否是 item.names 的前缀"""
        return is_names_start_with(tuple(r.item.names), tuple(r.focus.names))

    def to_names(self, r: MergedMatchResult) -> list[str]:
        """将菜单定位结果转换为菜单路径表示形式"""
        names = []

        # header匹配结果仅用于判断菜单是否打开
        if is_score_too_low(r.header):
            # 菜单未打开，返回空路径
            return []

        # focus匹配结果用于获取当前菜单标题
        if is_score_too_low(r.focus):
            # 菜单打开但无法定位标题，或者菜单未打开，返回空路径
            return []
        else:
            # 菜单打开且能定位标题
            names.append(r.focus.name)

        # list匹配结果用于获取当前菜单列表
        if is_score_too_low(r.list) or not self._is_list_belongs_to_focus(r):
            # 无法匹配已知列表，或不属于当前标题
            # 通常是刚切换到标题，列表还未激活或加载完成
            # 暂时继续往下尝试匹配条目
            pass

        # item匹配结果用于获取当前菜单条目
        if (
            is_score_too_low(r.item)
            or not self._is_item_belongs_to_list(r)
            or not self._is_item_belongs_to_focus(r)
        ):
            # 无法匹配已知条目，或不属于当前列表
            return names
        else:
            # 使用 item.names 作为最终路径
            return list(r.item.names)

        return names


class Action:
    """菜单导航动作"""

    TOGGLE_MENU = "toggle_menu"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    NAV_UP = "nav_up"
    NAV_DOWN = "nav_down"
    TAB_LEFT = "tab_left"
    TAB_RIGHT = "tab_right"


TABS_NUM = len(MENU_FOCUS_INFOS)
TABS_NAME_IDXS = {info["name"]: info["index"] for info in MENU_FOCUS_INFOS}
TABS_IDX_NAMES = {info["index"]: info["name"] for info in MENU_FOCUS_INFOS}

ItemType = Union[str, int]


def unify_tab(tab: ItemType) -> tuple[int, str]:
    """统一标签页表示形式

    :param tab: 标签页名称或索引

    :return: (标签页索引, 标签页名称)
    """
    if isinstance(tab, int):
        tab_idx = tab
        tab_name = TABS_IDX_NAMES.get(tab_idx)
    else:
        tab_name = tab
        tab_idx = TABS_NAME_IDXS.get(tab_name)
    return tab_idx, tab_name


def unify_item(item: ItemType, item_infos: list[dict]) -> tuple[int, str]:
    """统一菜单项表示形式

    :param item: 菜单项名称或索引
    :param item_infos: 该级菜单项信息列表

    :return: (菜单项索引, 菜单项名称)
    """
    item_idx_names = {info["index"]: info["name"] for info in item_infos}
    item_name_idxs = {info["name"]: info["index"] for info in item_infos}
    if isinstance(item, int):
        item_idx = item
        item_name = item_idx_names.get(item_idx)
    else:
        item_name = item
        item_idx = item_name_idxs.get(item_name)
    return item_idx, item_name


class MenuNavigatePlanner:
    """GTAV 菜单导航规划"""

    def _calc_tab_switch_actions(
        self, src_tab: Union[int, str], dst_tab: Union[int, str]
    ) -> tuple[str, int]:
        """计算标签页切换的动作

        :param src_tab: 当前标签页名称
        :param dst_tab: 目标标签页名称

        :return: 动作，格式为 (action, times) 元组
        """
        src_tab_idx, _ = unify_tab(src_tab)
        dst_tab_idx, _ = unify_tab(dst_tab)
        if src_tab_idx is None or dst_tab_idx is None:
            logger.warn(f"无法识别标签输入: {src_tab} 或 {dst_tab}")
            return None
        # 当前标签页即目标标签页
        if src_tab_idx == dst_tab_idx:
            return (None, 0)
        # 向右移动次数
        right_moves = (dst_tab_idx - src_tab_idx) % TABS_NUM
        # 向左移动次数
        left_moves = (src_tab_idx - dst_tab_idx) % TABS_NUM
        if right_moves <= left_moves:
            actions = (Action.TAB_RIGHT, right_moves)
        else:
            actions = (Action.TAB_LEFT, left_moves)
        return actions

    def _calc_item_nav_actions(
        self, src_item: ItemType, dst_item: ItemType, item_infos: list[dict]
    ) -> tuple[str, int]:
        """
        计算菜单项导航的动作

        :param src_item: 当前菜单项名称或索引
        :param dst_item: 目标菜单项名称或索引
        :param item_infos: 该级菜单项信息列表
        :return: 动作，格式为 (action, times) 元组
        """
        src_item_idx, _ = unify_item(src_item, item_infos)
        dst_item_idx, _ = unify_item(dst_item, item_infos)
        items_num = len(item_infos)
        if src_item_idx is None or dst_item_idx is None:
            logger.warn(f"无法识别菜单项输入: {src_item} 或 {dst_item}")
            return None
        # 当前菜单项即目标菜单项
        if src_item_idx == dst_item_idx:
            return (None, 0)
        # 向下移动次数
        down_moves = (dst_item_idx - src_item_idx) % items_num
        # 向上移动次数
        up_moves = (src_item_idx - dst_item_idx) % items_num
        if down_moves <= up_moves:
            actions = (Action.NAV_DOWN, down_moves)
        else:
            actions = (Action.NAV_UP, up_moves)
        return actions

    def plan_from_origin(self, names: list[str]) -> list[tuple[str, int]]:
        """规划从菜单关闭状态到 names 的导航路径

        :param names: 目标菜单项名称列表

        :return: 导航路径动作列表，每个动作为 (action, times) 元组
        """
        actions: list[tuple[str, int]] = []
        if not names:
            return []
        # 打开菜单
        actions.append((Action.TOGGLE_MENU, 1))
        # 切换到目标标签页
        dst_tab_name = names[0]
        tab_action = self._calc_tab_switch_actions(0, dst_tab_name)
        actions.append(tab_action)
        # 确认进入标签页
        actions.append((Action.CONFIRM, 1))
        # 依次导航到目标菜单项
        for level in range(2, len(names) + 1):
            pre_names = names[: level - 1]
            item_infos = MENU_PARENT_TO_ITEM_INFOS.get(tuple(pre_names))
            if item_infos is None:
                logger.warn(f"无法获取菜单项信息: 父级菜单路径 {pre_names}")
                break
            dst_item_name = names[level - 1]
            nav_action = self._calc_item_nav_actions(
                0, dst_item_name, item_infos=item_infos
            )
            actions.append(nav_action)
            # 确认进入菜单项
            actions.append((Action.CONFIRM, 1))
        return actions

    def plan(self, src_names: list[str], dst_names: list[str]) -> list[str]:
        """规划从 src_names 到 dst_names 的导航路径"""
        pass


class MenuNavigator:
    def __init__(self):
        self.interactor = MenuInteractor()
        self.locator = MenuLocatorRunner()
        self.capturer = ScreenCapturer()
        self.converter = LocateNamesConverter()
        self.planner = MenuNavigatePlanner()

    def locate(self) -> MergedMatchResult:
        """获取当前菜单定位结果"""
        frame_np = self.capturer.capture_frame().to_np()
        return self.locator.locate(frame_np)

    def locate_menu(self) -> list[str]:
        """获取当前菜单项名称列表"""
        locate_result = self.locate()
        return self.converter.to_names(locate_result)

    def ensure_menu_opened(self):
        """确保菜单处于打开状态"""
        result = self.locate()
        if is_score_too_low(result.header):
            # 菜单未打开，执行打开操作
            self.interactor.toggle_menu()

    def ensure_menu_closed(self):
        """确保菜单处于关闭状态"""
        result = self.locate()
        if is_score_high(result.header):
            # 菜单已打开，执行关闭操作
            self.interactor.toggle_menu()

    def go_to(self, names: list[str]) -> None:
        """导航到指定菜单项"""
        pass


def test_planner():
    logger.note("测试: MenuNavigatePlanner...")
    planner = MenuNavigatePlanner()

    def _log_planner(names: list[str]):
        logger.mesg(f"规划路径: {names}")
        paths = planner.plan_from_origin(names)
        logger.okay(f"规划结果: {paths}")

    names = ["设置"]
    _log_planner(names)

    names = ["在线", "差事", "进行差事"]
    _log_planner(names)

    names = ["在线", "差事", "进行差事", "已收藏的"]
    _log_planner(names)


def test_menu_navigator():
    logger.note("测试: MenuNavigator...")
    navigator = MenuNavigator()
    names = navigator.locate_menu()
    logger.mesg(f"当前路径: {names}")


if __name__ == "__main__":
    test_planner()
    # test_menu_navigator()

    # python -m gtaz.menus.navigates
