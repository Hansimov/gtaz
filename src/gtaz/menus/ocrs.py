"""
GTAV 菜单 OCR 模块

## 依赖安装

### 安装 rapidocr

- https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/

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

## 为什么不用 onnxruntime-gpu？

原因详见该贴：
- https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/24/onnxruntime-gpu推理/

结论：onnxruntime-gpu 版在动态输入情况下，推理速度要比CPU慢很多。

"""

from pathlib import Path
from tclogger import PathType, strf_path, TCLogger, logstr
from rapidocr import RapidOCR, EngineType
from rapidocr.utils.output import RapidOCROutput


logger = TCLogger(name="OCR", use_prefix=True, use_prefix_ms=True)

TORCH_GPU_PARAMS = {
    "Det.engine_type": EngineType.TORCH,
    "Cls.engine_type": EngineType.TORCH,
    "Rec.engine_type": EngineType.TORCH,
    "EngineConfig.torch.use_cuda": True,
    "EngineConfig.torch.gpu_id": 0,
}


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

    def __call__(self, path: PathType) -> RapidOCROutput:
        result = self.engine(
            strf_path(path),
            use_det=self.use_det,
            use_cls=self.use_cls,
            use_rec=self.use_rec,
        )
        return result


class OCREngineTester:
    def __init__(self):
        self.ocr = OCREngine(params=TORCH_GPU_PARAMS, use_cls=False)

    @staticmethod
    def _find_latest_jpg() -> str:
        menus_path = Path(__file__).parents[1] / "cache" / "menus"
        jpgs = list(menus_path.glob("**/*.jpg"))
        latest_jpg = max(jpgs, key=lambda p: p.stat().st_mtime)
        return strf_path(latest_jpg)

    def _log_elapsed(self, result: RapidOCROutput):
        elapses = result.elapse_list
        logger.okay(f"总耗时: {result.elapse:.3f}s")
        parts = ["文本检测", "方向分类", "文本识别"]
        elapses_str = ", ".join(
            f"{part}: {elapses[i] or 0:.3f}s" for i, part in enumerate(parts)
        )
        logger.mesg(f"各部分耗时: {elapses_str}")

    def _log_result(self, result: RapidOCROutput):
        logger.note(f"识别结果:")
        self._log_elapsed(result)

    def test(self):
        jpg = self._find_latest_jpg()

        logger.note(f"测试 OCR 模块...")
        logger.mesg(f"测试图像: {logstr.file(jpg)}")

        for i in range(10):
            result = self.ocr(jpg)
            if result:
                self._log_result(result)
            else:
                logger.warn(f"未能正常识别！")


if __name__ == "__main__":
    tester = OCREngineTester()
    tester.test()

    # Case: 测试 OCR 功能
    # python -m gtaz.menus.ocrs
