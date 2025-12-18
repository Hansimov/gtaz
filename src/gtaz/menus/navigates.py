"""GTAV 菜单导航模块"""

from dataclasses import dataclass
from tclogger import TCLogger

from .locates import MatchResult, MergedMatchResult
from .commons import MENU_FOCUS_INFOS, MENU_ITEM_INFOS
from .commons import is_names_start_with
from .interacts import MenuInteractor
from .locates import MATCH_THRESHOLD, MenuLocatorRunner
from ..screens import ScreenCapturer

logger = TCLogger(name="GTAVMenuNavigator", use_prefix=True, use_prefix_ms=True)


class LocateNamesConverter:
    @staticmethod
    def _is_score_too_low(result: MatchResult) -> bool:
        """判断匹配结果的分数是否低于阈值"""
        return result.score < MATCH_THRESHOLD

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
        if self._is_score_too_low(r.header):
            # 菜单未打开，返回空路径
            return []

        # focus匹配结果用于获取当前菜单标题
        if self._is_score_too_low(r.focus):
            # 菜单打开但无法定位标题，或者菜单未打开，返回空路径
            return []
        else:
            # 菜单打开且能定位标题
            names.append(r.focus.name)

        # list匹配结果用于获取当前菜单列表
        if self._is_score_too_low(r.list) or not self._is_list_belongs_to_focus(r):
            # 无法匹配已知列表，或不属于当前标题
            # 通常是刚切换到标题，列表还未激活或加载完成
            # 暂时继续往下尝试匹配条目
            pass

        # item匹配结果用于获取当前菜单条目
        if (
            self._is_score_too_low(r.item)
            or not self._is_item_belongs_to_list(r)
            or not self._is_item_belongs_to_focus(r)
        ):
            # 无法匹配已知条目，或不属于当前列表
            return names
        else:
            # 使用 item.names 作为最终路径
            return list(r.item.names)

        return names


class MenuNavigatePlanner:
    """GTAV 菜单导航规划"""

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

    def locate_menu(self) -> list[str]:
        """获取当前菜单项名称列表"""
        frame = self.capturer.capture_frame()
        frame_np = frame.to_np()
        locate_result = self.locator.locate(frame_np)
        locate_names = self.converter.to_names(locate_result)
        return locate_names

    def go_to(self, names: list[str]) -> None:
        """导航到指定菜单项"""
        pass


def test_planner():
    logger.note("测试: MenuNavigatePlanner...")
    planner = MenuNavigatePlanner()
    pass


def test_menu_navigator():
    logger.note("测试: MenuNavigator...")
    navigator = MenuNavigator()
    names = navigator.locate_menu()
    logger.mesg(f"当前路径: {names}")


if __name__ == "__main__":
    test_planner()
    # test_menu_navigator()

    # python -m gtaz.menus.navigates
