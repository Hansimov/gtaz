"""GTAV 画面分割与小地图提取"""

from pathlib import Path
from typing import Optional, Union
from PIL import Image
import numpy as np
from tclogger import TCLogger

logger = TCLogger(name="MinimapExtractor", use_prefix=True, use_prefix_ms=True)

# 获取当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR / "cache"
# 位置目录
LOCATIONS_DIR = CACHE_DIR / "locations"

# 参考分辨率和对应的小地图尺寸
# (画面宽度, 画面高度, 小地图宽度, 小地图高度)
RESOLUTION_MINIMAPS = [
    (1024, 768, 220, 160),
    (1366, 768, 240, 170),
    (1600, 900, 260, 190),
    (1904, 1001, 370, 220),  # 唯一一个小地图水平向右偏移的分辨率
    (1920, 1080, 330, 240),
]

# 默认图像质量（JPEG）
DEFAULT_QUALITY = 95


class MinimapExtractor:
    """
    GTAV 小地图提取器。

    从游戏截图或内存中的画面数据中提取左下角的小地图区域。
    根据画面分辨率动态计算小地图尺寸。
    """

    def __init__(
        self,
        quality: int = DEFAULT_QUALITY,
    ):
        """
        初始化小地图提取器。

        :param quality: 输出 JPEG 质量（1-100）
        """
        self.quality = max(1, min(100, quality))

    def _calculate_minimap_size(
        self, image_width: int, image_height: int
    ) -> tuple[int, int]:
        """
        根据画面分辨率动态计算小地图尺寸。

        如果完全匹配参考分辨率，直接返回对应尺寸；
        否则基于所有参考点进行多点平滑插值（反距离加权）。

        :param image_width: 画面宽度
        :param image_height: 画面高度
        :return: (小地图宽度, 小地图高度)
        """
        # 检查是否完全匹配
        for w, h, mw, mh in RESOLUTION_MINIMAPS:
            if image_width == w and image_height == h:
                return (mw, mh)

        # 多点平滑插值（反距离加权，Inverse Distance Weighting）
        # 使用画面宽度和高度的归一化距离
        total_weight = 0.0
        weighted_width = 0.0
        weighted_height = 0.0

        for w, h, mw, mh in RESOLUTION_MINIMAPS:
            # 计算归一化距离（宽度和高度各占一半权重）
            dw = (image_width - w) / 1000.0  # 归一化
            dh = (image_height - h) / 1000.0
            distance = (dw**2 + dh**2) ** 0.5

            # 避免除零，设置最小距离
            if distance < 0.001:
                return (mw, mh)

            # 反距离加权，使用平方反比使插值更平滑
            weight = 1.0 / (distance**2)
            total_weight += weight
            weighted_width += weight * mw
            weighted_height += weight * mh

        minimap_width = int(weighted_width / total_weight)
        minimap_height = int(weighted_height / total_weight)

        # 确保尺寸合理
        minimap_width = max(100, min(image_width, minimap_width))
        minimap_height = max(100, min(image_height, minimap_height))

        return (minimap_width, minimap_height)

    def _get_crop_region(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        """
        计算裁剪区域（左下角）。

        :param image_width: 原图宽度
        :param image_height: 原图高度
        :return: (left, top, right, bottom) 裁剪区域
        """
        minimap_width, minimap_height = self._calculate_minimap_size(
            image_width, image_height
        )
        left = 0
        top = image_height - minimap_height
        right = minimap_width
        bottom = image_height
        return (left, top, right, bottom)

    def extract_from_image(self, image: Image.Image) -> Image.Image:
        """
        从 PIL Image 对象中提取小地图。

        :param image: PIL Image 对象
        :return: 提取的小地图 Image 对象
        """
        img_width, img_height = image.size
        crop_region = self._get_crop_region(img_width, img_height)
        minimap = image.crop(crop_region)
        return minimap

    def extract_from_array(self, array: np.ndarray) -> np.ndarray:
        """
        从 numpy 数组中提取小地图。

        :param array: 图像数组，形状为 (height, width, channels)
        :return: 提取的小地图数组
        """
        img_height, img_width = array.shape[:2]
        left, top, right, bottom = self._get_crop_region(img_width, img_height)
        minimap = array[top:bottom, left:right]
        return minimap

    def extract_from_bytes(
        self,
        raw_data: bytes,
        width: int,
        height: int,
        mode: str = "BGRA",
    ) -> Image.Image:
        """
        从原始字节数据中提取小地图。

        :param raw_data: 原始位图数据
        :param width: 图像宽度
        :param height: 图像高度
        :param mode: 数据格式，默认 "BGRA"
        :return: 提取的小地图 Image 对象
        """
        image = Image.frombuffer("RGBA", (width, height), raw_data, "raw", mode, 0, 1)
        return self.extract_from_image(image)

    def extract_from_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Image.Image]:
        """
        从本地图片文件中提取小地图。

        :param input_path: 输入图片路径
        :param output_path: 输出图片路径（可选，如果提供则保存）
        :return: 提取的小地图 Image 对象，失败则返回 None
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.warn(f"输入文件不存在: {input_path}")
            return None

        try:
            image = Image.open(input_path)
            minimap = self.extract_from_image(image)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if minimap.mode == "RGBA":
                    minimap = minimap.convert("RGB")

                minimap.save(output_path, "JPEG", quality=self.quality)
                # logger.okay(f"小地图已保存: {output_path}")

            return minimap

        except Exception as e:
            logger.err(f"提取小地图失败 [{input_path}]: {e}")
            return None

    def batch_extract(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.jpg",
    ) -> list[Path]:
        """
        批量从目录中的图片文件提取小地图。

        :param input_dir: 输入目录
        :param output_dir: 输出目录（可选，如果不提供则自动生成）
        :param pattern: 文件匹配模式，默认 "*.jpg"
        :return: 成功保存的文件路径列表
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            logger.warn(f"输入目录不存在: {input_dir}")
            return []

        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_minimap"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(input_dir.glob(pattern))
        logger.mesg(f"找到 {len(input_files)} 个文件待处理")

        saved_files = []
        for input_file in input_files:
            output_file = output_dir / input_file.name
            result = self.extract_from_file(input_file, output_file)
            if result:
                saved_files.append(output_file)

        logger.okay(f"批量提取完成，成功处理 {len(saved_files)}/{len(input_files)} 个文件")
        return saved_files

    def process_all_floors(self, base_dir: Optional[Union[str, Path]] = None):
        """
        处理所有楼层目录（floor_*）中的截图。

        :param base_dir: 基础目录，默认为 cache/locations
        """
        if base_dir is None:
            base_dir = LOCATIONS_DIR
        else:
            base_dir = Path(base_dir)

        floor_dirs = sorted(
            [
                d
                for d in base_dir.glob("*floor_*")
                if d.is_dir() and not d.name.endswith("_minimap")
            ]
        )

        if not floor_dirs:
            logger.warn(f"未找到 floor_* 目录: {base_dir}")
            return

        logger.note(f"找到 {len(floor_dirs)} 个楼层目录")

        for floor_dir in floor_dirs:
            logger.note(f"处理目录: {floor_dir.name}")
            output_dir = base_dir / f"{floor_dir.name}_minimap"
            self.batch_extract(floor_dir, output_dir)

    def __repr__(self) -> str:
        return f"MinimapExtractor(quality={self.quality})"


def test_minimap_extractor():
    """测试小地图提取器。"""
    extractor = MinimapExtractor()
    logger.note(f"提取器信息: {extractor}")

    # 处理所有楼层
    extractor.process_all_floors()


if __name__ == "__main__":
    test_minimap_extractor()

    # python -m gtaz.segments
