"""
GTA 菜单 定位模块
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from tclogger import PathType, TCLogger, dict_to_str, logstr, strf_path
from typing import Literal, Union

from .commons import MENU_IMGS_DIR, LV1_INFOS, key_note, val_mesg


logger = TCLogger(name="GTAMenuLocator", use_prefix=True, use_prefix_ms=True)


PreprocessMode = Literal[
    "raw",
    "adaptive_bin",
    "text_map",
    "edge",
]


@dataclass(frozen=True)
class PreprocessConfig:
    mode: PreprocessMode = "text_map"
    blur_ksize: int = 3
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    morph_kernel: int = 7
    canny1: int = 60
    canny2: int = 150


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    if img_bgr.shape[2] == 1:
        return img_bgr[:, :, 0]
    # Lab 的 L 通道对亮度更稳定
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0]


def _odd_ksize(v: int, min_value: int = 3) -> int:
    v = int(v)
    if v < min_value:
        v = min_value
    if v % 2 == 0:
        v += 1
    return v


def _preprocess_gray(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    gray = gray.astype(np.uint8, copy=False)
    k = _odd_ksize(cfg.blur_ksize, 1)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if cfg.mode == "raw":
        return gray

    if cfg.mode == "adaptive_bin":
        bs = _odd_ksize(cfg.adaptive_block_size, 3)
        bin_img = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            int(cfg.adaptive_c),
        )
        return bin_img

    if cfg.mode == "text_map":
        ksize = max(3, int(cfg.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # 两种极性的文字统一到“亮响应”
        return cv2.max(tophat, blackhat)

    if cfg.mode == "edge":
        edges = cv2.Canny(gray, int(cfg.canny1), int(cfg.canny2))
        return edges

    raise ValueError(f"Unknown preprocess mode: {cfg.mode}")


def preprocess_image(img_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    gray = _to_gray(img_bgr)
    return _preprocess_gray(gray, cfg)


def score_polarity_textmap(gray_or_bgr: np.ndarray, morph_kernel: int = 7) -> dict:
    """计算文字极性分数（用于判断黑字白底 vs 白字黑底）。

    返回：
      - tophat_sum: 白字黑底更强
      - blackhat_sum: 黑字白底更强
      - polarity: 'white_on_dark' | 'dark_on_white'（基于 sum 比较的粗判）
    """
    gray = _to_gray(gray_or_bgr)
    ksize = max(3, int(morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    t_sum = float(np.sum(tophat))
    b_sum = float(np.sum(blackhat))
    polarity = "white_on_dark" if t_sum >= b_sum else "dark_on_white"
    return {
        "tophat_sum": t_sum,
        "blackhat_sum": b_sum,
        "polarity": polarity,
    }


def score_selected_by_contrast(gray_or_bgr: np.ndarray, text_mask: np.ndarray) -> float:
    """用背景-文字的亮度差做一个“选中反相”强度分数。

    - text_mask: 文字区域为 1/255，背景为 0。
    返回值越大，越像“亮背景 + 暗文字”（选中态）。
    """
    gray = _to_gray(gray_or_bgr).astype(np.float32)
    mask = (text_mask > 0).astype(np.uint8)
    if mask.sum() == 0 or mask.size == 0:
        return 0.0
    text_mean = float(gray[mask > 0].mean())
    bg_mean = float(gray[mask == 0].mean())
    return bg_mean - text_mean


def cv2_read(img_path: PathType) -> np.ndarray:
    """读取图像，支持中文路径。

    :param img_path: 图像路径

    :return: OpenCV 格式的图像数组
    """
    return cv2.imdecode(
        np.fromfile(strf_path(img_path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


class Lv1MenuMatcher:
    """Lv1菜单自适应匹配器"""

    def __init__(
        self,
        auto_scale: bool = True,
        ref_width: int = 1024,
        ref_height: int = 768,
        preprocess: PreprocessConfig | None = None,
    ):
        """初始化Lv1菜单匹配器。

        :param ref_height: 参考分辨率高度
        :param auto_scale: 使用自适应缩放
        """
        self.auto_scale = auto_scale
        self.ref_width = ref_width
        self.ref_height = ref_height
        self.preprocess = preprocess or PreprocessConfig()
        # 缓存模板的预处理结果：key=(template_name, scaled_w, scaled_h, preprocess)
        self._tpl_cache: dict[tuple[str, int, int, PreprocessConfig], np.ndarray] = {}

    def _is_same_size(self, img_np: np.ndarray) -> bool:
        """检查图像是否与参考尺寸相同。

        :param img_np: 输入图像
        :return: 是否相同尺寸
        """
        return img_np.shape[1] == self.ref_width and img_np.shape[0] == self.ref_height

    def _calc_scale(self, img_np: np.ndarray) -> float:
        """计算源图像相对于模板的缩放比例。

        :param img_np: 源图像
        :param template: 模板图像

        :return: 缩放比例
        """
        if not self.auto_scale:
            return 1.0

        # 使用图像高度计算缩放比例
        source_height = img_np.shape[0]
        scale = source_height / self.ref_height
        return scale

    def _scale_template(
        self, img_np: np.ndarray, template: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        """根据源图像尺寸缩放模板图像。"""
        if not self.auto_scale or self._is_same_size(img_np):
            return template, template.shape[1], template.shape[0]

        # 计算自适应缩放比例
        scale = self._calc_scale(img_np)

        # 根据缩放比例调整模板大小
        h, w = template.shape[:2]
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        # 检查缩放后的尺寸是否合理
        if (
            scaled_w > img_np.shape[1]
            or scaled_h > img_np.shape[0]
            or scaled_w < 10
            or scaled_h < 10
        ):
            # 如果尺寸不合理，使用原始模板
            scaled_template = template
            scaled_w, scaled_h = w, h
        else:
            # 缩放模板
            scaled_template = cv2.resize(
                template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
        return scaled_template, scaled_w, scaled_h

    def _get_template_feat(
        self,
        template_info: dict,
        scaled_template: np.ndarray,
        scaled_w: int,
        scaled_h: int,
    ) -> np.ndarray:
        key = (template_info["name"], scaled_w, scaled_h, self.preprocess)
        cached = self._tpl_cache.get(key)
        if cached is not None:
            return cached
        feat = preprocess_image(scaled_template, self.preprocess)
        self._tpl_cache[key] = feat
        return feat

    def match_template(self, img_np: np.ndarray, template_info: dict) -> dict:
        """对单个模板执行自适应匹配。

        :param img_np: 源图像
        :param template_info: 模板信息字典

        :return: 匹配结果字典
        """
        template = template_info["img"]

        # 自适应缩放模板
        scaled_template, scaled_w, scaled_h = self._scale_template(img_np, template)

        # 预处理到“结构域”匹配，提升对亮度/反相鲁棒性
        src_feat = preprocess_image(img_np, self.preprocess)
        tpl_feat = self._get_template_feat(
            template_info, scaled_template, scaled_w, scaled_h
        )

        # 执行模板匹配
        result = cv2.matchTemplate(src_feat, tpl_feat, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 计算匹配区域和中心点
        x, y = max_loc
        center_x = x + scaled_w // 2
        center_y = y + scaled_h // 2

        return {
            "name": template_info["name"],
            "confidence": float(max_val),
            "rect": (x, y, scaled_w, scaled_h),
            "center": (center_x, center_y),
            "level": template_info["level"],
            "index": template_info["index"],
        }


class Lv1MenuLocator:
    def __init__(self, threshold: float = 0.8):
        """初始化Lv1菜单定位器。

        :param threshold: 匹配阈值，范围 [0, 1]，默认 0.8
        """
        self.threshold = threshold
        self.templates = self._load_templates()
        # 默认使用 text_map：对“白字黑底 / 黑字白底”反相最友好
        self.matcher = Lv1MenuMatcher(preprocess=PreprocessConfig(mode="text_map"))

    def set_preprocess(self, cfg: PreprocessConfig) -> None:
        """切换匹配预处理配置（不破坏既有 API）。"""
        self.matcher.preprocess = cfg
        self.matcher._tpl_cache.clear()

    def _load_templates(self) -> list[dict]:
        """加载所有模板图像。

        :return: 包含模板信息的列表
        """
        templates = []
        for info in LV1_INFOS:
            template_path = MENU_IMGS_DIR / info["img"]
            template_img = cv2_read(template_path)
            templates.append(
                {
                    "name": info["name"],
                    "level": info["level"],
                    "index": info["index"],
                    "img": template_img,
                    "path": str(template_path),
                }
            )
        return templates

    def _load_image(self, img: Union[PathType, np.ndarray]) -> np.ndarray:
        """加载图像。

        :param img: 图像路径或 numpy 数组
        :return: OpenCV 格式的图像数组
        """
        if isinstance(img, np.ndarray):
            return img
        else:
            return cv2_read(img)

    def match_best(self, img: Union[PathType, np.ndarray]) -> dict:
        """匹配Lv1菜单，返回最佳匹配结果。

        :param img: 输入图像路径或数组

        :return: 匹配结果字典，包含以下键：
            - found: bool, 是否找到匹配
            - name: str, 匹配的菜单名称
            - confidence: float, 匹配置信度 [0, 1]
            - location: tuple, 匹配区域 (x, y, w, h)
            - center: tuple, 匹配区域中心点 (x, y)
        """
        source_img = self._load_image(img)

        best_match = {
            "found": False,
            "name": None,
            "confidence": 0.0,
            "rect": None,
            "center": None,
            "file": None,
        }

        for template_info in self.templates:
            match_result = self.matcher.match_template(source_img, template_info)

            # 如果当前匹配度更高，更新最佳匹配
            if match_result["confidence"] > best_match["confidence"]:
                best_match = match_result
                best_match["found"] = match_result["confidence"] >= self.threshold

        return best_match

    def match_all(
        self, img: Union[PathType, np.ndarray], threshold: float = None
    ) -> list[dict]:
        """匹配所有符合阈值的菜单项。

        :param img: 输入图像路径或数组
        :param threshold: 匹配阈值，如果为 None 则使用初始化时的阈值
        :return: 所有匹配结果的列表，按置信度降序排列
        """
        if threshold is None:
            threshold = self.threshold

        img_np = self._load_image(img)
        matches = []

        for template_info in self.templates:
            match_result = self.matcher.match_template(img_np, template_info)

            if match_result["confidence"] >= threshold:
                match_result["found"] = True
                matches.append(match_result)

        # 按置信度降序排列
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches


class MenuLocatorTester:
    def __init__(self):
        self.locator = Lv1MenuLocator(threshold=0.8)

    def _plot_result_on_image(self, img: np.ndarray, result: dict) -> np.ndarray:
        """在图像上绘制匹配结果。

        :param img: 输入图像数组
        :param results: 匹配结果

        :return: 绘制后的图像数组
        """
        x, y, w, h = result["rect"]
        confidence = result["confidence"]
        name = result["name"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{confidence:.2f}"
        cv2.putText(
            img,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (122, 122, 0),
            2,
        )
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

    def _save_result_json(self, result: dict, save_path: Path) -> None:
        """保存JSON结果。

        :param result: 匹配结果字典
        :param save_path: JSON保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        # logger.file(f"  * 信息已保存: {save_path}")

    def _log_result(self, result: dict):
        """打印匹配结果

        :param result: 匹配结果字典
        """
        logger.note("匹配结果:")
        info_dict = {
            "found": result["found"],
            "name": result["name"],
            "confidence": f"{result['confidence']:.4f}",
            "rect": result["rect"],
            "center": result["center"],
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _log_result_line(self, result: dict, idx: int = None):
        if idx is not None:
            idx_str = f"[{idx}] "
        else:
            idx_str = ""
        conf_str = f"{result['confidence']:.4f}"
        logger.mesg(
            f"  * {idx_str}"
            f"{logstr.okay(result['name'])}, "
            f"{key_note('置信度')}: {logstr.okay(conf_str)}, "
            f"{key_note('区域')}: {val_mesg(result['rect'])}"
        )

    def _log_results(self, results: list[dict]):
        """打印多个匹配结果

        :param results: 匹配结果列表
        """
        logger.okay(f"共匹配到 {len(results)} 个结果:")
        for idx, result in enumerate(results, 1):
            self._log_result_line(result, idx=idx)

    def test_match_best(self, img_path: PathType) -> dict:
        """测试单个最佳匹配。"""
        logger.note(f"测试图像: {img_path}")
        result = self.locator.match_best(img_path)
        self._log_result(result)
        return result

    def test_match_all(self, img_path: str, threshold: float = 0.8):
        """测试所有匹配。"""
        logger.note(f"测试图像: {img_path}")
        logger.note(f"匹配阈值: {threshold}")
        results = self.locator.match_all(img_path, threshold=threshold)
        self._log_results(results)
        return results

    def test_match_and_visualize(
        self, img_path: PathType, idx: int = None
    ) -> np.ndarray:
        """可视化匹配结果。

        :param img_path: 输入图像路径
        """
        img_path = Path(img_path)
        source_img = cv2_read(img_path)
        result = self.locator.match_best(str(img_path))
        self._log_result_line(result, idx=idx)

        source_img = self._plot_result_on_image(source_img, result)
        save_path = self._get_result_path(img_path)
        self._save_result_image(source_img, save_path)
        json_path = save_path.with_suffix(".json")
        self._save_result_json(result, json_path)
        return source_img

    def batch_test_match_and_visualize(self, img_dir: PathType):
        """批量可视化测试目录中的所有图像。

        :param img_dir: 图像目录
        """
        logger.note("运行批量可视化测试...")

        img_dir = Path(img_dir)
        img_paths = sorted(img_dir.glob("*.jpg"))

        logger.note(f"读取图像目录: [{img_dir}]")
        logger.okay(f"找到 {len(img_paths)} 张图像")

        for i, img_path in enumerate(img_paths, 1):
            idx_str = f"[{logstr.mesg(i)}/{logstr.file(len(img_paths))}] "
            logger.note(f"{idx_str}处理图像: {img_path.name}")
            self.test_match_and_visualize(str(img_path))

        logger.okay(f"批量测试完成！共测试图像: {len(img_paths)}")

    def run(self):
        """运行所有测试"""
        logger.note("=" * 50)
        logger.note("菜单定位测试")
        logger.note("=" * 50)

        cache_menus = Path(__file__).parents[1] / "cache" / "menus"
        # img_dir = cache_menus / "2025-12-14_23-01-58"
        img_dir = cache_menus / "2025-12-15_08-22-57"

        # imgs = list(img_dir.glob("*.jpg"))
        # img = imgs[0]
        # self.test_match_and_visualize(str(img))

        self.batch_test_match_and_visualize(img_dir)


def test_menu_locator():
    tester = MenuLocatorTester()
    tester.run()


if __name__ == "__main__":
    test_menu_locator()

    # python -m gtaz.menus.locates
