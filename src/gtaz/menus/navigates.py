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
        return is_names_start_with(r.list.names, r.focus.names)

    @staticmethod
    def _is_item_belongs_to_list(r: MergedMatchResult) -> bool:
        """判断 list.names 是否是 item.names 的前缀"""
        return is_names_start_with(r.item.names, r.list.names)

    @staticmethod
    def _is_item_belongs_to_focus(r: MergedMatchResult) -> bool:
        """判断focus.names 是否是 item.names 的前缀"""
        return is_names_start_with(r.item.names, r.focus.names)

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

    def sum_actions_times(self, actions: list[tuple[str, int]]) -> int:
        """计算动作列表中所有动作的总次数"""
        total = 0
        for action, times in actions:
            if action and times:
                total += times
        return total

    @staticmethod
    def _log_error_parent_names(parent_names: list[str]) -> None:
        logger.warn(f"无法获取菜单项信息: 父级菜单路径 {parent_names}")

    def _calc_common_prefix_length(
        self, src_names: list[str], dst_names: list[str]
    ) -> int:
        """找到两个路径的公共前缀长度

        :param src_names: 当前菜单项名称列表
        :param dst_names: 目标菜单项名称列表

        :return: 公共前缀长度
        """
        common_len = 0
        min_len = min(len(src_names), len(dst_names))
        for i in range(min_len):
            if src_names[i] == dst_names[i]:
                common_len += 1
            else:
                break
        return common_len

    def _plan_backward(
        self, src_names: list[str], dst_names: list[str]
    ) -> list[tuple[str, int]]:
        """规划回退到目标路径

        :param src_names: 当前菜单项名称列表
        :param dst_names: 公共前缀名称列表

        :return: 回退动作列表
        """
        actions: list[tuple[str, int]] = []
        # 回退到 common_len 层级
        cancel_times = len(src_names) - len(dst_names)
        if cancel_times > 0:
            actions.append((Action.CANCEL, cancel_times))
        return actions

    def _plan_backward_to_sibling(
        self, src_names: list[str], dst_names: list[str]
    ) -> list[tuple[str, int]]:
        """规划回退到平级位置

        :param src_names: 当前菜单项名称列表
        :param dst_names: 公共前缀名称列表

        :return: 回退动作列表
        """
        actions: list[tuple[str, int]] = []
        # 回退到 common_len + 1 层级，也即平级位置
        cancel_times = len(src_names) - len(dst_names) - 1
        if cancel_times > 0:
            actions.append((Action.CANCEL, cancel_times))
        return actions

    def _plan_forward(
        self, src_names: list[str], dst_names: list[str]
    ) -> list[tuple[str, int]]:
        """规划从公共前缀位置前进到目标位置

        :param src_names: 公共前缀名称列表
        :param dst_names: 目标菜单项名称列表

        :return: 前进动作列表
        """
        actions: list[tuple[str, int]] = []
        if src_names == dst_names:
            return []
        # 从公共前缀的下一级开始，逐级前进到目标
        for level in range(len(src_names) + 1, len(dst_names) + 1):
            # 确认进入下一级菜单
            actions.append((Action.CONFIRM, 1))
            # 获取当前层级的父级路径和映射信息
            parent_names = dst_names[: level - 1]
            item_infos = MENU_PARENT_TO_ITEM_INFOS.get(tuple(parent_names))
            if item_infos is None:
                self._log_error_parent_names(parent_names)
                break
            # 导航到目标条目
            dst_name = dst_names[level - 1]
            nav_action = self._calc_item_nav_actions(0, dst_name, item_infos=item_infos)
            actions.append(nav_action)
        return actions

    def _is_sibling(self, src_names: list[str], dst_names: list[str]) -> bool:
        """判断当前路径和目标路径是否平级

        也即：路径长度相同，且有公共父级，只有最后一项不同
        例如: ['在线', '差事'] -> ['在线', '游玩清单']

        :param src_names: 当前菜单项名称列表
        :param dst_names: 目标菜单项名称列表
        :param common_len: 公共前缀长度

        :return: 是否为平级切换
        """
        common_len = self._calc_common_prefix_length(src_names, dst_names)
        return (
            len(src_names) == len(dst_names)  # 路径长度相同
            and len(src_names) > 1  # 至少有一级父级
            and common_len == len(src_names) - 1  # 只有最后一项不同
        )

    def _plan_sibling(
        self, src_names: list[str], dst_names: list[str]
    ) -> list[tuple[str, int]]:
        """规划平级切换的导航路径

        :param src_names: 当前菜单项名称列表
        :param dst_names: 目标菜单项名称列表

        :return: 导航路径动作列表
        """
        actions: list[tuple[str, int]] = []
        if src_names == dst_names:
            return []
        # 获取父级路径
        parent_names = src_names[:-1]
        # 获取该级的条目信息
        item_infos = MENU_PARENT_TO_ITEM_INFOS.get(tuple(parent_names))
        if item_infos is None:
            logger.warn(f"无法获取菜单项信息: 父级菜单路径 {parent_names}")
            return actions
        # 获取当前和目标条目名称
        src_item_name = src_names[-1]
        dst_item_name = dst_names[-1]
        # 计算从当前项到目标项的导航动作
        nav_action = self._calc_item_nav_actions(
            src_item_name, dst_item_name, item_infos=item_infos
        )
        actions.append(nav_action)
        return actions

    def plan_from_origin(self, dst_names: list[str]) -> list[tuple[str, int]]:
        """从头规划导航路径：从菜单关闭状态，到 names 对应位置

        :param dst_names: 目标菜单项名称列表

        :return: 导航路径动作列表，每个动作为 (action, times) 元组
        """
        actions: list[tuple[str, int]] = []
        if not dst_names:
            return []
        # 打开菜单
        actions.append((Action.TOGGLE_MENU, 1))
        # 切换到目标标签页
        dst_tab_name = dst_names[0]
        tab_action = self._calc_tab_switch_actions(0, dst_tab_name)
        actions.append(tab_action)
        # 前进到目标位置
        forward_actions = self._plan_forward([dst_tab_name], dst_names)
        actions.extend(forward_actions)
        return actions

    def plan_from_source(
        self, src_names: list[str], dst_names: list[str]
    ) -> list[tuple[str, int]]:
        """规划导航路径: 从 src_names 到 dst_names

        :param src_names: 当前菜单项名称列表
        :param dst_names: 目标菜单项名称列表

        :return: 导航路径动作列表，每个动作为 (action, times) 元组
        """
        # 边界情况: 当前菜单未打开，从头规划
        if not src_names:
            return self.plan_from_origin(dst_names)
        # 边界情况: 目标是关闭菜单，直接关闭菜单
        if not dst_names:
            return [(Action.TOGGLE_MENU, 1)]
        # 边界情况: 当前位置即目标位置，什么都不做
        if src_names == dst_names:
            return []

        actions: list[tuple[str, int]] = []

        # 找到公共前缀
        common_len = self._calc_common_prefix_length(src_names, dst_names)
        common_names = src_names[:common_len]

        # 特殊情况：跨标签页
        if common_len == 0:
            tab_action = self._calc_tab_switch_actions(src_names[0], dst_names[0])
            actions.append(tab_action)
            # 从标签页切换后，前进到目标位置
            forward_actions = self._plan_forward([dst_names[0]], dst_names)
            actions.extend(forward_actions)
            return actions

        # 从当前位置回退到公共前缀下一级，也即平级位置
        if len(src_names) >= common_len + 1:
            backward_actions = self._plan_backward_to_sibling(src_names, common_names)
            actions.extend(backward_actions)

        # 使用公共前缀下一级，作为中间位置
        mid_names = src_names[: common_len + 1]
        mid_len = len(mid_names)

        # 目标恰好为公共前缀，回退
        if len(dst_names) < mid_len:
            # 事实上等价于 len(dst_names) == common_len
            backward_actions = self._plan_backward(mid_names, dst_names)
            actions.extend(backward_actions)

        # 目标在平级或者更深层级，平移
        if len(dst_names) >= mid_len:
            sibling_actions = self._plan_sibling(mid_names, dst_names[:mid_len])
            actions.extend(sibling_actions)

        # 目标为中间位置的更深层级，前进
        if len(dst_names) > mid_len:
            forward_actions = self._plan_forward(mid_names, dst_names)
            actions.extend(forward_actions)

        return actions


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

    def locate_names(self) -> list[str]:
        """获取当前菜单项名称列表"""
        locate_result = self.locate()
        return self.converter.to_names(locate_result)

    def plan_actions(self, dst_names: list[str]) -> list[tuple[str, int]]:
        """规划导航到指定菜单项的动作"""
        src_names = self.locate_names()
        actions = self.planner.plan_from_source(src_names, dst_names)
        return actions

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

    def execute_actions(self, actions: list[tuple[str, int]]):
        for action in actions:
            act, times = action
            if not act or not times:
                continue
            elif act == Action.TOGGLE_MENU:
                self.interactor.toggle_menu()
            elif act == Action.CONFIRM:
                self.interactor.confirm()
            elif act == Action.CANCEL:
                self.interactor.cancel(times)
            elif act == Action.NAV_UP:
                self.interactor.nav_up(times)
            elif act == Action.NAV_DOWN:
                self.interactor.nav_down(times)
            elif act == Action.TAB_LEFT:
                self.interactor.tab_left(times)
            elif act == Action.TAB_RIGHT:
                self.interactor.tab_right(times)
            else:
                logger.warn(f"未知导航动作: {act}")
            # 每次操作后等待菜单稳定
            self.interactor.wait_until_ready()

    def _log_retry_go_to(self, names: list[str], retry: int, max_retries: int) -> None:
        logger.warn(f"[{retry}/{max_retries}] 未导航到预期位置，当前路径: {names}")

    def go_to(self, dst_names: list[str], max_retries: int = 3) -> list[str]:
        """导航到指定菜单项"""
        retry = 0
        while retry < max_retries:
            actions = self.plan_actions(dst_names)
            logger.mesg(f"导航动作: {actions}")
            self.execute_actions(actions)
            names = self.locate_names()
            if names == dst_names:
                return names
            else:
                retry += 1
                self._log_retry_go_to(names, retry, max_retries)
                continue
        return names


def test_planner():
    logger.note("测试: MenuNavigatePlanner...")
    planner = MenuNavigatePlanner()

    def _log_plan_from_origin(names: list[str]):
        logger.mesg(f"规划路径 (从原点): {names}")
        paths = planner.plan_from_origin(names)
        logger.okay(f"规划结果: {paths}")
        logger.mesg(f"总步数: {planner.sum_actions_times(paths)}\n")

    def _log_plan_from_source(src_names: list[str], dst_names: list[str]):
        logger.mesg(f"规划路径: {src_names} -> {dst_names}")
        paths = planner.plan_from_source(src_names, dst_names)
        logger.okay(f"规划结果: {paths}")
        logger.mesg(f"总步数: {planner.sum_actions_times(paths)}\n")

    # 测试从原点出发的路径规划
    logger.note("=== 测试: 从原点出发 ===")
    origin_cases = [
        ["设置"],
        ["在线", "差事", "进行差事"],
        ["在线", "差事", "进行差事", "已收藏的"],
    ]
    for names in origin_cases:
        _log_plan_from_origin(names)

    # 测试完整路径规划
    logger.note("=== 测试: 边界情况 ===")
    boundary_cases = [
        ([], ["在线"]),  # 空路径到目标
        (["在线"], []),  # 当前位置到空路径 (关闭菜单)
        (["在线"], ["在线"]),  # 当前位置即目标位置
    ]
    for src, dst in boundary_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 同级标签页切换 ===")
    tab_switch_cases = [
        (["在线"], ["设置"]),
        (["设置"], ["地图"]),
    ]
    for src, dst in tab_switch_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 同一标签页内的导航 ===")
    same_tab_nav_cases = [
        (["在线"], ["在线", "差事"]),  # 一级到二级
        (["在线"], ["在线", "退出游戏"]),  # 一级到二级
        (["在线", "差事"], ["在线", "差事", "进行差事"]),  # 二级到三级
        (
            ["在线", "差事", "进行差事"],
            ["在线", "差事", "进行差事", "已收藏的"],
        ),  # 三级到四级
        (["在线"], ["在线", "差事", "进行差事", "已收藏的"]),  # 一级到四级
    ]
    for src, dst in same_tab_nav_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 同一标签页内的回退 ===")
    backward_cases = [
        (
            ["在线", "差事", "进行差事", "已收藏的"],
            ["在线", "差事", "进行差事"],
        ),  # 四级到三级
        (["在线", "差事", "进行差事"], ["在线", "差事"]),  # 三级到二级
        (["在线", "差事", "进行差事", "已收藏的"], ["在线"]),  # 四级到一级
    ]
    for src, dst in backward_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 同一标签页内的平级切换 ===")
    sibling_switch_cases = [
        (["在线", "差事"], ["在线", "游玩清单"]),  # 优化: 只需要导航
        (
            ["在线", "差事", "快速加入"],
            ["在线", "差事", "进行差事"],
        ),  # 优化: 只需要导航
    ]
    for src, dst in sibling_switch_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 跨标签页的复杂导航 ===")
    cross_tab_cases = [
        (["在线", "差事", "进行差事", "已收藏的"], ["设置"]),  # 优化: 无需回退
        (["设置"], ["在线", "差事", "进行差事", "已收藏的"]),  # 优化: 无需回退
        (["在线", "差事", "进行差事"], ["地图"]),  # 优化: 无需回退
    ]
    for src, dst in cross_tab_cases:
        _log_plan_from_source(src, dst)

    logger.note("=== 测试: 不同分支的切换 ===")
    branch_switch_cases = [
        (
            ["在线", "差事", "进行差事"],
            ["在线", "寻找新战局"],
        ),
        (
            ["在线", "差事", "进行差事", "已收藏的"],
            ["在线", "游玩清单"],
        ),
        (
            ["在线", "加入帮会成员"],
            ["在线", "差事", "进行差事", "已收藏的"],
        ),
    ]
    for src, dst in branch_switch_cases:
        _log_plan_from_source(src, dst)


def test_menu_navigator():
    logger.note("测试: MenuNavigator...")
    navigator = MenuNavigator()
    names = navigator.locate_names()
    logger.mesg(f"当前路径: {names}")

    dst_names = ["在线", "差事", "进行差事", "已收藏的", "夺取"]
    # dst_names = ["在线"]
    logger.mesg(f"导航到: {dst_names}")
    navigator.go_to(dst_names)

    logger.mesg(f"当前路径: {names}")


if __name__ == "__main__":
    # test_planner()
    test_menu_navigator()

    # python -m gtaz.menus.navigates
