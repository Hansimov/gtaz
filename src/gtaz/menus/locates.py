"""
GTA 菜单 定位模块
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from tclogger import PathType, TCLogger, dict_to_str, logstr, strf_path
from typing import Literal, Union

from .commons import MENU_IMGS_DIR
from .commons import MENU_HEADER_INFOS, MENU_FOCUS_INFOS
from .commons import MENU_LIST_INFOS, MENU_ITEM_INFOS
from .commons import REF_WIDTH, REF_HEIGHT, MAX_LIST_SIZE
from .commons import key_note, val_mesg, is_names_start_with


logger = TCLogger(name="GTAMenuLocator", use_prefix=True, use_prefix_ms=True)


MATCH_THRESHOLD = 0.5
HIGH_THRESHOLD = 0.8


def cv2_read(img_path: PathType) -> np.ndarray:
    """读取图像，支持中文路径。

    :param img_path: 图像路径

    :return: OpenCV 格式的图像数组
    """
    return cv2.imdecode(
        np.fromfile(strf_path(img_path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def load_img(img: Union[PathType, np.ndarray]) -> np.ndarray:
    """加载图像。

    :param img: 图像路径或 numpy 数组

    :return: OpenCV 格式的图像数组
    """
    if isinstance(img, np.ndarray):
        return img
    else:
        return cv2_read(img)


def crop_img(img_np: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return img_np[y1:y2, x1:x2]


FeatureType = Literal["raw", "header", "focus", "list", "item"]


@dataclass
class TemplateInfo:
    """模板信息"""

    name: str
    names: tuple[str, ...]  # 仅限: 列表、条目
    parent: tuple[str, ...]  # 仅限: 列表、条目
    img: np.ndarray
    img_path: str
    level: int
    index: int = -1  # 仅限: 标题、焦点、条目
    total: int = -1  # 仅限: 列表


@dataclass
class MatchResult:
    """模板匹配结果"""

    name: str
    names: tuple[str, ...]
    score: float
    rect: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    rect_size: tuple[int, int]  # (scaled_w, scaled_h)
    rect_center: tuple[int, int]  # (center_x, center_y)
    level: int
    index: int = -1  # 对于列表模板，使用 -1 表示无效索引
    total: int = -1  # 对于列表模板，存储总条目数

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


class FailedMatchResult(MatchResult):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            names=None,
            score=0.0,
            rect=(0, 0, 0, 0),
            rect_size=(0, 0),
            rect_center=(0, 0),
            level=-1,
            index=-1,
            total=-1,
        )


@dataclass
class MergedMatchResult:
    """合并的模板匹配结果"""

    header: MatchResult = None
    focus: MatchResult = None
    list: MatchResult = None
    item: MatchResult = None

    def __post_init__(self):
        """初始化默认值"""
        if self.header is None:
            self.header = FailedMatchResult("默认标题")
        if self.focus is None:
            self.focus = FailedMatchResult("默认焦点")
        if self.list is None:
            self.list = FailedMatchResult("默认列表")
        if self.item is None:
            self.item = FailedMatchResult("默认条目")

    def to_dict(self) -> dict:
        """转换为字典"""
        res_dict = {}
        for key in ["header", "focus", "list", "item"]:
            result = getattr(self, key)
            if result is None:
                res_dict[key] = None
            else:
                res_dict[key] = result.to_dict()
        return res_dict


@dataclass(frozen=True)
class FeatureExtractorConfig:
    """特征提取器配置"""

    mode: FeatureType = "text_map"
    blur_ksize: int = 3
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    morph_kernel: int = 7
    canny1: int = 60
    canny2: int = 150


class ImageFeatureExtractor:
    """图像特征提取器，将图像转换为适合模板匹配的特征图"""

    def __init__(self, config: FeatureExtractorConfig = None):
        self.config = config or FeatureExtractorConfig()
        self.extractors = {
            "raw": self.extract_raw,
            "header": self.extract_text_map,
            "focus": self.extract_raw,
            "list": self.extract_text_map,
            "item": self.extract_raw,
        }

    def get_hash(self) -> int:
        """获取配置的哈希值"""
        return hash(
            (
                self.config.mode,
                self.config.blur_ksize,
                self.config.adaptive_block_size,
                self.config.adaptive_c,
                self.config.morph_kernel,
                self.config.canny1,
                self.config.canny2,
            )
        )

    @staticmethod
    def to_gray(img_bgr: np.ndarray) -> np.ndarray:
        """将BGR图像转换为灰度图。使用Lab色彩空间的L通道以提高亮度稳定性。"""
        if img_bgr.ndim == 2:
            return img_bgr
        if img_bgr.shape[2] == 1:
            return img_bgr[:, :, 0]
        # Lab 的 L 通道对亮度更稳定
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    @staticmethod
    def ensure_odd_ksize(value: int, min_value: int = 3) -> int:
        """确保内核大小为奇数（OpenCV要求）"""
        value = max(int(value), min_value)
        return value if value % 2 == 1 else value + 1

    def apply_blur(self, gray: np.ndarray) -> np.ndarray:
        """应用高斯模糊以降低噪声"""
        ksize = self.ensure_odd_ksize(self.config.blur_ksize, 1)
        if ksize > 1:
            return cv2.GaussianBlur(gray, (ksize, ksize), 0)
        return gray

    def extract_raw(self, gray: np.ndarray) -> np.ndarray:
        """原始模式：仅返回灰度图"""
        return gray

    def extract_adaptive_bin(self, gray: np.ndarray) -> np.ndarray:
        """自适应二值化模式：适用于光照不均匀的场景"""
        block_size = self.ensure_odd_ksize(self.config.adaptive_block_size, 3)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )

    def extract_text_map(self, gray: np.ndarray) -> np.ndarray:
        """文字映射模式：提取文字特征，对黑底白字/白底黑字反相鲁棒"""
        ksize = max(3, int(self.config.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # 统一两种极性的文字到"亮响应"
        return cv2.max(tophat, blackhat)

    def extract_edge(self, gray: np.ndarray) -> np.ndarray:
        """边缘检测模式：提取轮廓特征"""
        return cv2.Canny(gray, self.config.canny1, self.config.canny2)

    def extract_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """执行完整的特征提取流程

        :param img_bgr: 输入的BGR图像
        :return: 提取后的特征图
        """
        # 转为灰度图
        gray = self.to_gray(img_bgr)
        gray = gray.astype(np.uint8, copy=False)
        if self.config.mode not in ["focus", "item"]:
            # 应用模糊
            gray = self.apply_blur(gray)
        # 根据模式，选择特征提取策略
        extractor = self.extractors.get(self.config.mode)
        return extractor(gray)

    def score_polarity_textmap(self, img: np.ndarray, morph_kernel: int = 7) -> dict:
        """计算文字极性分数（用于判断黑字白底 vs 白字黑底）。

        :return: 包含以下键的字典：
            - tophat_sum: 白字黑底更强
            - blackhat_sum: 黑字白底更强
            - polarity: 'white_on_dark' | 'dark_on_white'（基于 sum 比较的粗判）
        """
        gray = self.to_gray(img)
        ksize = max(3, int(morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        t_sum = float(np.sum(tophat))
        b_sum = float(np.sum(blackhat))
        if t_sum >= b_sum:
            polarity = "white_on_dark"
        else:
            polarity = "dark_on_white"
        return {
            "tophat_sum": t_sum,
            "blackhat_sum": b_sum,
            "polarity": polarity,
        }

    def score_selected_by_contrast(
        self, img: np.ndarray, text_mask: np.ndarray
    ) -> float:
        """用背景-文字的亮度差做一个"选中反相"强度分数。

        :param img: 输入图像
        :param text_mask: 文字区域为 1/255，背景为 0。

        :return: 亮度差分数，数值越大表示反相越明显
        """
        gray = self.to_gray(img).astype(np.float32)
        mask = (text_mask > 0).astype(np.uint8)
        if mask.sum() == 0 or mask.size == 0:
            return 0.0
        text_mean = float(gray[mask > 0].mean())
        bg_mean = float(gray[mask == 0].mean())
        return bg_mean - text_mean


class MenuMatcher:
    """菜单匹配器"""

    def __init__(
        self,
        auto_scale: bool = True,
        ref_width: int = REF_WIDTH,
        ref_height: int = REF_HEIGHT,
    ):
        """初始化菜单匹配器。

        :param auto_scale: 使用自适应缩放
        :param ref_width: 参考分辨率宽度
        :param ref_height: 参考分辨率高度
        """
        self.auto_scale = auto_scale
        self.ref_width = ref_width
        self.ref_height = ref_height
        # 当前图像尺寸
        self.img_width: int = ref_width
        self.img_height: int = ref_height
        # 创建特征提取器，不同元素使用不同的配置
        self.header_extractor = ImageFeatureExtractor(
            FeatureExtractorConfig(mode="header")
        )
        self.focus_extractor = ImageFeatureExtractor(
            FeatureExtractorConfig(mode="focus")
        )
        self.list_extractor = ImageFeatureExtractor(FeatureExtractorConfig(mode="list"))
        self.item_extractor = ImageFeatureExtractor(FeatureExtractorConfig(mode="item"))
        # 缓存模板的特征图
        self._feature_cache: dict[tuple, np.ndarray] = {}

    def set_img_size(self, img_np: np.ndarray) -> None:
        """设置当前要处理的图像尺寸。

        :param img_np: 输入图像
        """
        self.img_height, self.img_width = img_np.shape[:2]

    def _is_same_size(self) -> bool:
        """检查当前图像是否与参考尺寸相同。

        :return: 是否相同尺寸
        """
        return self.img_width == self.ref_width and self.img_height == self.ref_height

    def _calc_scale(self) -> float:
        """计算当前图像相对于模板的缩放比例。

        :return: 缩放比例
        """
        if not self.auto_scale:
            return 1.0
        # 使用图像高度计算缩放比例
        return self.img_height / self.ref_height

    def _is_bad_scale_size(self, scaled_w: int, scaled_h: int):
        return (
            scaled_w > self.img_width
            or scaled_h > self.img_height
            or scaled_w < 8
            or scaled_h < 8
        )

    def _scale_template(self, template: np.ndarray) -> tuple[np.ndarray, int, int]:
        """根据当前图像尺寸缩放模板图像。"""
        if not self.auto_scale or self._is_same_size():
            return template, template.shape[1], template.shape[0]
        # 计算自适应缩放比例
        scale = self._calc_scale()
        # 根据缩放比例调整模板大小
        h, w = template.shape[:2]
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        if self._is_bad_scale_size(scaled_w, scaled_h):
            # 如果尺寸不合理，使用原始模板
            scaled_template = template
            scaled_w, scaled_h = w, h
        else:
            # 缩放模板
            scaled_template = cv2.resize(
                template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
        return scaled_template, scaled_w, scaled_h

    def _get_template_features(
        self,
        template_info: TemplateInfo,
        scaled_template: np.ndarray,
        scaled_w: int,
        scaled_h: int,
        extractor: ImageFeatureExtractor,
    ) -> np.ndarray:
        """获取模板特征图（带缓存）"""
        key = (
            template_info.name,
            template_info.img_path,
            scaled_w,
            scaled_h,
            extractor.get_hash(),
        )
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        features = extractor.extract_features(scaled_template)
        self._feature_cache[key] = features
        return features

    def _is_bad_img_size(self, img_np: np.ndarray) -> bool:
        return img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0

    def _is_bad_template_size(
        self, img_np: np.ndarray, scaled_w: int, scaled_h: int
    ) -> bool:
        src_h, src_w = img_np.shape[:2]
        return scaled_h > src_h or scaled_w > src_w

    @staticmethod
    def _build_match_result(
        result: np.ndarray,
        template_info: TemplateInfo,
        scaled_w: int,
        scaled_h: int,
    ) -> MatchResult:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        x1, y1 = max_loc
        x2 = x1 + scaled_w
        y2 = y1 + scaled_h
        center_x = x1 + scaled_w // 2
        center_y = y1 + scaled_h // 2
        return MatchResult(
            name=template_info.name,
            names=template_info.names,
            score=float(max_val),
            rect=(x1, y1, x2, y2),
            rect_size=(scaled_w, scaled_h),
            rect_center=(center_x, center_y),
            level=template_info.level,
            index=template_info.index,
        )

    def match_template(
        self,
        img_np: np.ndarray,
        template_info: TemplateInfo,
        extractor: ImageFeatureExtractor,
    ) -> MatchResult:
        """对单个模板执行自适应匹配。

        :param img_np: 源图像
        :param template_info: 模板信息
        :param extractor: 特征提取器

        :return: 匹配结果 MatchResult
        """
        # 检查源图像是否为空
        if self._is_bad_img_size(img_np):
            return FailedMatchResult("无效图像")
        # 自适应缩放模板
        template = template_info.img
        scaled_template, scaled_w, scaled_h = self._scale_template(template)
        # 检查模板尺寸是否大于源图像
        if self._is_bad_template_size(img_np, scaled_w, scaled_h):
            return FailedMatchResult("无效模板")
        # 提取图像和模板特征
        src_features = extractor.extract_features(img_np)
        tpl_features = self._get_template_features(
            template_info, scaled_template, scaled_w, scaled_h, extractor
        )
        # 执行模板匹配
        result = cv2.matchTemplate(src_features, tpl_features, cv2.TM_CCOEFF_NORMED)
        return self._build_match_result(
            result=result,
            template_info=template_info,
            scaled_w=scaled_w,
            scaled_h=scaled_h,
        )


class MenuLocator:
    def __init__(self, threshold: float = 0.65):
        """初始化菜单定位器。

        :param threshold: 匹配阈值，范围 [0, 1]
        """
        self.threshold = threshold
        self.matcher = MenuMatcher()
        self._load_templates()

    @staticmethod
    def _build_template_info(info: dict) -> TemplateInfo:
        """从配置字典加载单个模板信息。

        :param info: 模板配置字典
        :param kwargs: 额外的字段（如 index 或 total）
        :return: TemplateInfo 实例
        """
        img_path = MENU_IMGS_DIR / info["img"]
        template_img = cv2_read(img_path)
        return TemplateInfo(
            name=info["name"],
            names=info.get("names", None),
            parent=info.get("parent", None),
            img=template_img,
            img_path=str(img_path),
            level=info["level"],
            index=info.get("index", -1),
            total=info.get("total", -1),
        )

    def _load_templates(self) -> None:
        """加载所有模板图像。"""
        # 加载标题模板
        used_header_names = ["地图", "在线", "设置"]
        self.header_templates: list[TemplateInfo] = [
            self._build_template_info(info)
            for info in MENU_HEADER_INFOS
            if info["name"] in used_header_names
        ]
        # 加载焦点模板
        self.focus_templates: list[TemplateInfo] = [
            self._build_template_info(info) for info in MENU_FOCUS_INFOS
        ]
        # 加载列表模板
        self.list_templates: list[TemplateInfo] = [
            self._build_template_info(info) for info in MENU_LIST_INFOS
        ]
        # 加载条目模板
        self.item_templates: list[TemplateInfo] = [
            self._build_template_info(info) for info in MENU_ITEM_INFOS
        ]

    @staticmethod
    def _should_update_best_match(
        match_result: MatchResult, best_match: MatchResult
    ) -> bool:
        return best_match is None or match_result.score > best_match.score

    @staticmethod
    def _is_bad_match_result(match_result: MatchResult) -> bool:
        return match_result is None or match_result.score == 0.0

    def match_header(self, img_np: np.ndarray) -> MatchResult:
        """匹配菜单，返回最匹配的菜单标题区域。

        :param img: 输入图像路径或数组

        :return: 匹配结果 MatchResult，包含以下字段：
            - name: str, 匹配的菜单名称
            - score: float, 匹配置信度 [0, 1]
            - rect: tuple, 匹配区域 (x1, y1, x2, y2)
            - rect_size: tuple, 缩放后的模板尺寸 (scaled_w, scaled_h)
            - rect_center: tuple, 匹配区域中心点 (center_x, center_y)
            - level: int, 菜单层级
            - index: int, 菜单索引
        """
        self.matcher.set_img_size(img_np)
        best_match = None
        for template_info in self.header_templates:
            match_result = self.matcher.match_template(
                img_np, template_info, self.matcher.header_extractor
            )
            if self._should_update_best_match(match_result, best_match):
                best_match = match_result
        return best_match

    def _align_result(
        self, result: MatchResult, offset_x: int, offset_y: int
    ) -> MatchResult:
        """将局部坐标的匹配结果转换为全局坐标。

        :param result: 局部坐标的匹配结果
        :param offset_x: X轴偏移量
        :param offset_y: Y轴偏移量
        :return: 全局坐标的匹配结果
        """
        # 匹配区域的局部坐标
        lx1, ly1, lx2, ly2 = result.rect
        # 匹配区域的全局坐标
        gx1 = offset_x + lx1
        gx2 = offset_x + lx2
        gy1 = offset_y + ly1
        gy2 = offset_y + ly2
        # 匹配区域的全局中心
        center_gx = result.rect_center[0] + offset_x
        center_gy = result.rect_center[1] + offset_y
        # 修正坐标后的最终结果
        result.rect = (gx1, gy1, gx2, gy2)
        result.rect_size = result.rect_size
        result.rect_center = (center_gx, center_gy)
        return result

    def _align_focus_result(
        self, focus_result: MatchResult, header_result: MatchResult
    ) -> MatchResult:
        # 标题栏区域的全局坐标：hx1, hy1, hx2, hy2
        hx1, hy1, hx2, hy2 = header_result.rect
        return self._align_result(focus_result, hx1, hy1)

    def match_focus(
        self, img_np: np.ndarray, header_result: MatchResult
    ) -> MatchResult:
        """匹配菜单，返回最匹配的菜单焦点区域。

        :param img_np: 完整的输入图像数组
        :param header_result: match_header 的匹配结果

        :return: 匹配结果 MatchResult，坐标为全局坐标（相对于完整图像）
        """
        cropped_img = crop_img(img_np, header_result.rect)
        best_match = None
        for template_info in self.focus_templates:
            match_result = self.matcher.match_template(
                cropped_img, template_info, self.matcher.focus_extractor
            )
            if self._should_update_best_match(match_result, best_match):
                best_match = match_result
        final_match = self._align_focus_result(best_match, header_result)
        return final_match

    def _calc_list_region(
        self, header_result: MatchResult
    ) -> tuple[int, int, int, int]:
        """计算列表的匹配区域。

        列表区域的左上角位置是 header_result 的左下角，
        大小为 MAX_LIST_SIZE * scale，其中 scale 是全局图像缩放比例。

        :param header_result: 标题匹配结果
        :return: 列表区域 (x1, y1, x2, y2)
        """
        hx1, hy1, hx2, hy2 = header_result.rect
        # 使用全局缩放比例，而不是基于 header 高度计算
        scale = self.matcher._calc_scale()
        # 计算列表区域大小
        list_w = int(MAX_LIST_SIZE[0] * scale)
        list_h = int(MAX_LIST_SIZE[1] * scale)
        # 列表左上角 = 标题左下角
        list_x1 = hx1
        list_y1 = hy2
        list_x2 = list_x1 + list_w
        list_y2 = list_y1 + list_h
        # 确保不超出图像边界
        list_x2 = min(list_x2, self.matcher.img_width)
        list_y2 = min(list_y2, self.matcher.img_height)
        return (list_x1, list_y1, list_x2, list_y2)

    def _filter_list_templates(self, focus_result: MatchResult) -> list[TemplateInfo]:
        """筛选列表模板，仅保留 names 以 focus_result.name 开头的模板。

        :param focus_result: 焦点匹配结果

        :return: 筛选后的列表模板列表
        """
        filtered = [
            template
            for template in self.list_templates
            if is_names_start_with(template.names, (focus_result.name,))
        ]
        return filtered

    def match_list(
        self, img_np: np.ndarray, header_result: MatchResult, focus_result: MatchResult
    ) -> MatchResult:
        """匹配列表区域。

        :param img_np: 完整的输入图像数组
        :param header_result: 标题匹配结果
        :param focus_result: 焦点匹配结果
        :return: 列表匹配结果，坐标为全局坐标
        """
        # 计算列表区域
        list_region = self._calc_list_region(header_result)
        list_x1, list_y1, list_x2, list_y2 = list_region
        # 裁剪列表区域
        cropped_img = crop_img(img_np, list_region)
        # 匹配模板
        filtered_list_templates = self._filter_list_templates(focus_result)
        best_match = None
        for template_info in filtered_list_templates:
            match_result = self.matcher.match_template(
                cropped_img, template_info, self.matcher.list_extractor
            )
            if self._should_update_best_match(match_result, best_match):
                best_match = match_result
        # 无效匹配
        if self._is_bad_match_result(best_match):
            return FailedMatchResult("未知列表")
        # 将局部坐标转换为全局坐标
        final_match = self._align_result(best_match, list_x1, list_y1)
        return final_match

    def _filter_item_templates(
        self, focus_result: MatchResult, list_result: MatchResult
    ) -> list[TemplateInfo]:
        """筛选条目模板，仅保留 names 同时以 focus_result.name 和 list_result.names 开头，并且正好在 list_result.names 下一个层级的模板。

        :param focus_result: 焦点匹配结果
        :param list_result: 列表匹配结果
        :return: 筛选后的条目模板列表
        """
        list_prefix = list_result.names
        filtered = [
            template
            for template in self.item_templates
            if is_names_start_with(template.names, (focus_result.name,))
            and is_names_start_with(template.names, list_prefix)
            and len(template.names) == len(list_prefix) + 1
        ]
        return filtered

    def match_item(
        self,
        img_np: np.ndarray,
        header_result: MatchResult,
        focus_result: MatchResult,
        list_result: MatchResult,
    ) -> MatchResult:
        """匹配条目区域。

        :param img_np: 完整的输入图像数组
        :param header_result: 标题匹配结果
        :param focus_result: 焦点匹配结果
        :param list_result: 列表匹配结果
        :return: 条目匹配结果，坐标为全局坐标
        """
        # 获取列表区域
        list_rect = list_result.rect
        list_x1, list_y1, list_x2, list_y2 = list_rect
        # 裁剪列表区域
        cropped_img = crop_img(img_np, list_rect)
        # 匹配模板
        filtered_item_templates = self._filter_item_templates(focus_result, list_result)
        best_match = None
        for template_info in filtered_item_templates:
            match_result = self.matcher.match_template(
                cropped_img, template_info, self.matcher.item_extractor
            )
            if self._should_update_best_match(match_result, best_match):
                best_match = match_result
        # 无效匹配
        if self._is_bad_match_result(best_match):
            return FailedMatchResult("未知条目")
        # 将局部坐标转换为全局坐标
        final_match = self._align_result(best_match, list_x1, list_y1)
        return final_match


def is_score_too_low(result: MatchResult, threshold: float = None) -> bool:
    """判断匹配结果的分数是否低于阈值"""
    return result.score < (threshold or MATCH_THRESHOLD)


def is_score_high(result: MatchResult, threshold: float = None) -> bool:
    """判断匹配结果的分数是否高于阈值"""
    return result.score >= (threshold or HIGH_THRESHOLD)


class MenuLocatorRunner:
    def __init__(self):
        self.locator = MenuLocator()

    def _plot_result_on_image(
        self,
        img: np.ndarray,
        result: MatchResult,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """在图像上绘制匹配结果。

        :param img: 输入图像数组
        :param result: 匹配结果
        :param color: 矩形颜色 (R, G, B)
        :param thickness: 线条粗细

        :return: 绘制后的图像数组
        """
        x1, y1, x2, y2 = result.rect
        bgr_color = (color[2], color[1], color[0])  # 转为 BGR 格式
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, thickness)
        return img

    def _get_result_path(self, img_path: Path) -> Path:
        """获取可视化结果的输出路径。

        :param img_path: 输入图像路径

        :return: 输出路径
        """
        # 获取父目录名称，添加 _locates 后缀
        parent_dir = img_path.parent
        parent_name = parent_dir.name
        locates_dir = parent_dir.parent / f"{parent_name}_locates"
        locates_dir.mkdir(parents=True, exist_ok=True)
        return locates_dir / img_path.name

    def _save_result_image(self, img: np.ndarray, save_path: Path) -> None:
        """保存可视化图像。

        :param img: 图像数组
        :param save_path: 图像保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        if is_success:
            im_buf_arr.tofile(str(save_path))
        # logger.file(f"  * 绘制已保存: {save_path}")

    def _save_result_json(
        self, result: Union[MatchResult, MergedMatchResult], save_path: Path
    ) -> None:
        """保存JSON结果。

        :param result: 匹配结果
        :param save_path: JSON保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        # logger.file(f"  * 信息已保存: {save_path}")

    def _log_result(self, result: MatchResult):
        """打印匹配结果

        :param result: 匹配结果
        """
        logger.note("匹配结果:")
        info_dict = {
            "name": result.name,
            "score": f"{result.score:.4f}",
            "rect": result.rect,
            "rect_size": result.rect_size,
            "rect_center": result.rect_center,
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _log_result_line(
        self, result: MatchResult, idx: int = None, name_type: str = None
    ):
        if idx is None:
            idx_str = ""
        else:
            idx_str = f"[{idx}] "

        if name_type is None:
            name_type_str = ""
        else:
            name_type_str = f"{key_note(name_type)}: "

        score = result.score
        if score >= 0.88:
            logstr_func = logstr.okay
        elif score < 0.5:
            logstr_func = logstr.warn
        else:
            logstr_func = logstr.file

        name_str = logstr_func(result.name)
        score_str = logstr_func(f"{score:.4f}")
        rect_str = logstr_func(result.rect)

        logger.mesg(
            f"  * {idx_str}"
            f"{name_type_str}{name_str}, "
            f"{key_note('置信度')}: {score_str}, "
            f"{key_note('区域')}: {rect_str}"
        )

    def locate(self, img_np: np.ndarray, verbose: bool = True) -> MergedMatchResult:
        """匹配所有菜单元素，返回合并的匹配结果。"""
        # 匹配标题
        header_result = self.locator.match_header(img_np)
        if verbose:
            self._log_result_line(header_result, name_type="标题")
        # 标题匹配分数过低，提前返回
        if is_score_too_low(header_result):
            return MergedMatchResult(header=header_result)
        # 匹配焦点
        focus_result = self.locator.match_focus(img_np, header_result=header_result)
        if verbose:
            self._log_result_line(focus_result, name_type="焦点")
        # 焦点匹配分数过低，提前返回
        if is_score_too_low(focus_result):
            return MergedMatchResult(header=header_result, focus=focus_result)
        # 匹配列表
        list_result = self.locator.match_list(
            img_np, header_result=header_result, focus_result=focus_result
        )
        if verbose:
            self._log_result_line(list_result, name_type="列表")
        # 匹配条目
        item_result = self.locator.match_item(
            img_np,
            header_result=header_result,
            focus_result=focus_result,
            list_result=list_result,
        )
        if verbose:
            self._log_result_line(item_result, name_type="条目")
        # 合并结果
        merged_result = MergedMatchResult(
            header=header_result, focus=focus_result, list=list_result, item=item_result
        )
        return merged_result

    def locate_and_visualize(self, img_path: PathType) -> np.ndarray:
        """可视化匹配结果。

        :param img_path: 输入图像路径
        """
        # 读取图像
        img_path = Path(img_path)
        img_np = cv2_read(img_path)
        # 匹配
        merged_result = self.locate(img_np)
        header_result = merged_result.header
        focus_result = merged_result.focus
        list_result = merged_result.list
        item_result = merged_result.item
        # 可视化
        img_np = self._plot_result_on_image(img_np, header_result, color=(0, 255, 0))
        img_np = self._plot_result_on_image(img_np, focus_result, color=(255, 0, 0))
        img_np = self._plot_result_on_image(img_np, list_result, color=(0, 128, 128))
        img_np = self._plot_result_on_image(img_np, item_result, color=(128, 128, 0))
        # 保存可视化结果和匹配信息
        save_path = self._get_result_path(img_path)
        self._save_result_image(img_np, save_path)
        json_path = save_path.with_suffix(".json")
        self._save_result_json(merged_result, json_path)
        return img_np

    def multi_locate_and_visualize(self, img_dir: PathType):
        """批量可视化测试目录中的所有图像。

        :param img_dir: 图像目录
        """
        logger.note("运行批量可视化处理...")
        # 读取目录中所有图像
        img_dir = Path(img_dir)
        img_paths = sorted(img_dir.glob("*.jpg"))
        logger.note(f"读取图像目录: [{img_dir}]")
        logger.okay(f"找到 {len(img_paths)} 张图像")
        # 逐张处理
        for i, img_path in enumerate(img_paths, 1):
            idx_str = f"[{logstr.mesg(i)}/{logstr.file(len(img_paths))}] "
            logger.note(f"{idx_str}处理图像: {img_path.name}")
            self.locate_and_visualize(str(img_path))
        logger.okay(f"批量处理完成！共处理图像: {len(img_paths)}")

    def run(self):
        """运行所有测试"""
        logger.note("=" * 50)
        logger.note("菜单定位测试")
        logger.note("=" * 50)

        cache_menus = Path(__file__).parents[1] / "cache" / "menus"
        # img_dir = cache_menus / "2025-12-14_23-01-58"
        # img_dir = cache_menus / "2025-12-15_08-22-57"
        img_dir = cache_menus / "2025-12-17_09-50-09"

        # imgs = list(img_dir.glob("*.jpg"))
        # img = imgs[0]
        # self.match_and_visualize(str(img))

        self.multi_locate_and_visualize(img_dir)


def run_menu_locator():
    runner = MenuLocatorRunner()
    runner.run()


if __name__ == "__main__":
    run_menu_locator()

    # python -m gtaz.menus.locates
