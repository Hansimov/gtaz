# GTAZ
Visual-based AI for real-time tasks in GTAV.

![](https://img.shields.io/pypi/v/gtaz?label=gtaz&color=blue&cacheSeconds=60)

## Thanks to

Some ideas and codes (like `GamepadSimulator` ) are inspired by and inherited from:

- [shibeta/JNTMbot_python](https://github.com/shibeta/JNTMbot_python)

Without the original author's great work, this project would not be possible.

## Install and Setup

TBD

## Demos

### 事务所自动走到任务点

行为克隆 + resnet18 + tensorrt：

https://github.com/user-attachments/assets/b8f11ecf-4cc4-4458-8905-debdf7b7a654

### 识别角色楼层

根据小地图，识别角色在事务所的楼层。训练日志：

![recognizes_v2](./img/recognize_v2_train.png)

交叉验证：

![recognizes_v2_validation](./img/recognize_v2_validation.png)

## Developer Notes

See [Developer Notes](./DEV.md) for details of data collection, model training, runtime exporting, and real-time inference.