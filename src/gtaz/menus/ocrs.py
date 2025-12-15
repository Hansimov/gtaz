"""
GTAV 菜单 OCR 模块

## 依赖安装

### 安装 rapidocr

- https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install

```sh
pip install rapidocr
```

### 安装 PyTorch

Windows下安装CUDA版本需要指定 `--index-url`:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 安装 onnxruntime

CPU版本：

```sh
pip install onnxruntime
```

GPU版本：
```sh
# 卸载安装的CPU版本和旧的GPU版本，重新安装GPU版本
pip uninstall onnx onnxruntime onnxscript onnxruntime-gpu -y
pip install --upgrade onnx onnxscript onnxruntime-gpu
```

## 快速测试

```sh
rapidocr -img "<image_full_path>" --vis_res
```

## 用法和参数

使用教程：
- https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage

默认配置文件：
- https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml

参数：
- https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters

## 为什么不用 onnxruntime-gpu？

原因详见该贴：
- https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/24/onnxruntime-gpu推理

结论：onnxruntime-gpu 版在动态输入情况下，推理速度要比CPU慢很多。

"""

import numpy as np

from tclogger import PathType, TCLogger, logstr, int_bits
from typing import Union

from PIL import Image
from PIL.Image import Image as PImage
from rapidocr import RapidOCR, EngineType
from rapidocr.utils.output import RapidOCROutput

from .commons import find_latest_jpg, key_note, val_mesg

logger = TCLogger(name="OCR", use_prefix=True, use_prefix_ms=True)


TORCH_GPU_PARAMS = {
    "Det.engine_type": EngineType.TORCH,
    "Cls.engine_type": EngineType.TORCH,
    "Rec.engine_type": EngineType.TORCH,
    "EngineConfig.torch.use_cuda": True,
    "EngineConfig.torch.gpu_id": 0,
}


class PillowImager:
    def load(self, path: PathType) -> PImage:
        """加载图像文件。

        :param path: 图像文件路径

        :return: PIL Image 对象
        """
        with Image.open(path) as img:
            return img.copy()

    def resize(
        self, img: PImage, w: int = None, h: int = None, ratio: float = None
    ) -> PImage:
        """调整图像尺寸。
        - 按指定的宽高缩放，若另一维度为 None，则等比缩放。
        - 如果 ratio 不为 None，则按比例缩放。
        - 仅能指定宽高或比例中的一种。

        :param img: 输入图像（PImage）
        :param w: 目标宽度
        :param h: 目标高度
        :param ratio: 缩放比例

        :return: 调整尺寸后的图像（PImage）
        """
        orig_w, orig_h = img.size

        if ratio is not None:
            # 按比例缩放
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
        elif w is not None and h is not None:
            # 指定宽高
            new_w = w
            new_h = h
        elif w is not None:
            # 仅指定宽度，等比缩放
            new_w = w
            new_h = int(orig_h * w / orig_w)
        elif h is not None:
            # 仅指定高度，等比缩放
            new_h = h
            new_w = int(orig_w * h / orig_h)
        else:
            # 无参数，返回原图
            return img

        return img.resize((new_w, new_h))

    def to_np(self, img: PImage) -> np.ndarray:
        """将 PIL Image 转换为 numpy 数组。

        :param img: PIL Image 对象

        :return: numpy 数组
        """
        return np.array(img)


class OCREngine:
    def __init__(
        self,
        params: dict = None,
        use_det: bool = True,
        use_cls: bool = True,
        use_rec: bool = True,
    ):
        self.engine = RapidOCR(params=params)
        self.use_det = use_det
        self.use_cls = use_cls
        self.use_rec = use_rec
        self.imager = PillowImager()

    def __call__(self, img: Union[PathType, np.ndarray]) -> RapidOCROutput:
        result = self.engine(
            img,
            use_det=self.use_det,
            use_cls=self.use_cls,
            use_rec=self.use_rec,
        )
        return result


class TxtBoxSorter:
    @staticmethod
    def _sort_by_dist(
        txts: tuple[str], boxes: np.ndarray
    ) -> list[tuple[str, np.ndarray, float]]:
        """根据文本框左上角到原点的距离对识别结果进行排序（从小到大）。

        :param txts: 识别的文本内容
        :param boxes: 文本框坐标

        :return: 排序后的 (文本, 文本框, 距离) 列表
        """
        # 计算每个文本框左上角到原点的距离
        distances = []
        for box in boxes:
            x, y = box[0]
            distance = np.sqrt(x**2 + y**2)
            distances.append(round(distance))
        # 按距离从小到大排序
        sorted_indices = np.argsort(distances)
        sorted_items = [(txts[i], boxes[i], distances[i]) for i in sorted_indices]
        return sorted_items

    @staticmethod
    def _sort_by_area(
        txts: tuple[str], boxes: np.ndarray
    ) -> list[tuple[str, np.ndarray, float]]:
        """根据文本框面积对识别结果进行排序（从大到小）。

        :param txts: 识别的文本内容
        :param boxes: 文本框坐标

        :return: 排序后的 (文本, 文本框, 面积) 列表
        """
        # 计算每个文本框的面积
        areas = []
        for box in boxes:
            # box 是 4 个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # 使用 Shoelace 公式计算多边形面积
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            area = 0.5 * np.abs(
                np.dot(x_coords, np.roll(y_coords, 1))
                - np.dot(y_coords, np.roll(x_coords, 1))
            )
            areas.append(round(area))
        # 按面积从大到小排序
        sorted_indices = np.argsort([-area for area in areas])
        sorted_items = [(txts[i], boxes[i], areas[i]) for i in sorted_indices]
        return sorted_items

    @staticmethod
    def _sort_by_width(
        txts: tuple[str], boxes: np.ndarray
    ) -> list[tuple[str, np.ndarray, float]]:
        """根据文本框宽度对识别结果进行排序（从大到小）。

        :param txts: 识别的文本内容
        :param boxes: 文本框坐标

        :return: 排序后的 (文本, 文本框, 宽度) 列表
        """
        # 计算每个文本框的宽度
        widths = []
        for box in boxes:
            # box 是 4 个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # 计算左上角到右上角的距离作为宽度
            x_coords = box[:, 0]
            width = max(x_coords) - min(x_coords)
            widths.append(round(width))
        # 按宽度从大到小排序
        sorted_indices = np.argsort([-w for w in widths])
        sorted_items = [(txts[i], boxes[i], widths[i]) for i in sorted_indices]
        return sorted_items

    @staticmethod
    def _sort_by_height(
        txts: tuple[str], boxes: np.ndarray
    ) -> list[tuple[str, np.ndarray, float]]:
        """根据文本框高度对识别结果进行排序（从大到小）。

        :param txts: 识别的文本内容
        :param boxes: 文本框坐标

        :return: 排序后的 (文本, 文本框, 高度) 列表
        """
        # 计算每个文本框的高度
        heights = []
        for box in boxes:
            # box 是 4 个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # 计算左上角到左下角的距离作为高度
            y_coords = box[:, 1]
            height = max(y_coords) - min(y_coords)
            heights.append(round(height))
        # 按高度从大到小排序
        sorted_indices = np.argsort([-h for h in heights])
        sorted_items = [(txts[i], boxes[i], heights[i]) for i in sorted_indices]
        return sorted_items


SORT_METHODS = {
    "距离": TxtBoxSorter._sort_by_dist,
    "面积": TxtBoxSorter._sort_by_area,
    "宽度": TxtBoxSorter._sort_by_width,
    "高度": TxtBoxSorter._sort_by_height,
}


class OCREngineTester:
    def __init__(self):
        self.ocr = OCREngine(params=TORCH_GPU_PARAMS, use_cls=False)
        self.imager = PillowImager()

    def _load_image_as_np(self, path: PathType) -> np.ndarray:
        img = self.imager.load(path)
        # img = self.imager.resize(img, h=384)
        img_np = self.imager.to_np(img)
        return img_np

    def _log_elapses(self, result: RapidOCROutput):
        logger.okay(f"总耗时: {result.elapse:.3f}s")
        parts = ["文本检测", "方向分类", "文本识别"]
        elapses = result.elapse_list
        elapses_str = ", ".join(
            f"{part}: {elapses[i] or 0:.3f}s" for i, part in enumerate(parts)
        )
        logger.mesg(f"各部分耗时: {elapses_str}")

    def _log_txts(self, result: RapidOCROutput):
        logger.note("文本内容:")
        txts = result.txts
        boxes = result.boxes
        scores = result.scores
        sort_name = "高度"
        sorted_txts_boxes = SORT_METHODS[sort_name](txts, boxes)
        idx_len = int_bits(len(sorted_txts_boxes))
        sort_value_len = int_bits(
            max(sort_value for _, _, sort_value in sorted_txts_boxes)
        )
        for i, (txt, box, sort_value) in enumerate(sorted_txts_boxes, 1):
            old_idx = list(result.txts).index(txt)
            score = scores[old_idx]
            sort_val_str = f"{sort_value:>{sort_value_len}}"
            logger.mesg(
                f"- [{i:>{idx_len}}] "
                f"{key_note('置信度')}: {score:.3f}, "
                f"{key_note(sort_name)}: {sort_val_str}, "
                f"{key_note('文本')}: {logstr.okay(txt)}"
            )

    def _log_result(self, result: RapidOCROutput):
        logger.note(f"识别结果:")
        with logger.temp_indent(2):
            self._log_elapses(result)
            self._log_txts(result)

    def test(self):
        img_path = find_latest_jpg()

        logger.note(f"测试 OCR 模块...")
        logger.mesg(f"测试图像: {logstr.file(img_path)}")

        # img_np = self._load_image_as_np(img_path)
        # result = self.ocr(img_np)

        result = self.ocr(img_path)
        if result:
            self._log_result(result)
        else:
            logger.warn(f"未能正常识别！")


if __name__ == "__main__":
    tester = OCREngineTester()
    tester.test()

    # Case: 测试 OCR 功能
    # python -m gtaz.menus.ocrs
