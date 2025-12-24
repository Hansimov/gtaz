"""音频模板匹配模块"""

import argparse
import json
import numpy as np
import time

from dataclasses import dataclass
from pathlib import Path
from tclogger import TCLogger, logstr, dict_to_lines

# 当前模块所在目录
MODULE_DIR = Path(__file__).parent
# 缓存目录
CACHE_DIR = MODULE_DIR.parent / "cache"

# 模板目录
TEMPLATES_DIR = MODULE_DIR / "wavs"
# 模板文件正则表达式
TEMPLATE_REGEX = r"templatex_.*\.wav"

# 测试音频目录
SOUNDS_DIR = CACHE_DIR / "sounds"
# 测试结果参考文件
REF_JSON = SOUNDS_DIR / "refs.json"
# 测试结果输出文件
REF_JSON = SOUNDS_DIR / "results.json"

# 特征X轴样本点数
FEATURE_X_POINTS = 20
# 特征Y轴样本点数
FEATURE_Y_POINTS = 256

# 特征匹配窗口（毫秒）
MATCH_WINDOW_MS = 500
# 特征匹配步长（毫秒）
MATCH_STEP_MS = 150
# 特征匹配阈值
MATCH_GATE = 0.6
# 候选者之间最小相邻时间间隔（毫秒）
CANDIDATE_MIN_OFFSET_MS = 300


logger = TCLogger(
    name="AudioTemplateMatcher",
    use_prefix=False,
    use_file=True,
    file_path=Path(__file__).parent / "output.log",
    file_mode="w",
)


class TemplateLoader:
    """音频模板加载"""

    pass


class AudioDataSamplesUnifier:
    """音频数据统一化"""

    pass


@dataclass
class Feature:
    """音频特征数据类"""

    pass


class FeatureExtractor:
    """音频特征提取"""

    pass


@dataclass
class MatchCandidate:
    """音频模板匹配候选数据类"""

    pass


@dataclass
class MatchResult:
    """音频模板匹配结果数据类"""

    pass


class FeatureMatcher:
    """音频特征匹配"""

    pass


class TestDataSamplesLoader:
    """测试数据加载和预处理"""

    pass


class FeatureMatchTester:
    """音频特征匹配测试"""

    pass


class TemplateMatcherArgParser:
    pass


if __name__ == "__main__":
    pass

    # python -m gtaz.audios.signals_v3 --test
