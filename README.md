# GTAZ
Visual-based AI for real-time tasks in GTAV.

![](https://img.shields.io/pypi/v/gtaz?label=gtaz&color=blue&cacheSeconds=60)

## 当前功能

- [x] 大仓自动取货
- [x] 事务所自动走到“别惹德瑞”任务点

## 安装和使用

- 大仓自动取货
  - See: [自动取货模块 - 安装和使用教程](./AUTO_PICKUP.md)

- 事务所自动走到“别惹德瑞”任务点
  - TBD

## Demos

### 大仓自动取货

TBD

### 事务所自动走到任务点

行为克隆 + resnet18 + tensorrt：

https://github.com/user-attachments/assets/b8f11ecf-4cc4-4458-8905-debdf7b7a654

## 致谢

没有他们的无私工作，这个项目不可能诞生。

### shibeta

一切都要从下面这个项目开始。很多初始的想法和代码，都是直接启发和继承自它：

- [shibeta/JNTMbot_python](https://github.com/shibeta/JNTMbot_python)

这个项目最初是用于德瑞差事 Bot。但是 [shibeta](https://github.com/shibeta) 的工作让我意识到：

**Python + 手柄，可以把所有自动化操作 GTAV 的工作都放在后台。**

这个手柄的思路当然肯定很多人在更早之前就已经想到和用上了，但是 shibeta 是第一次让我注意到并且能快速复用的。

### 傲弗拉

[傲弗拉](https://space.bilibili.com/2913798) 的这两个教程，帮我节省了不少摸索原理的时间。他在评论区的解答，也在我 debug 时提供了很多帮助：

* 【GTA教程】我要当老板！全自动大仓员工无限取货
  * https://www.bilibili.com/video/BV1DdoMYdEic

* 【GTA教程】Trueboss可能是目前最好用的自动大仓工具
  * https://www.bilibili.com/video/BV1aNm3BvEMb

Trueboss 插件下载：
* 插件链接：
  * https://jobtp.lanzouw.com/s/Trueboss
* 使用文档：
  * https://docs.qq.com/doc/DVFNMaUZQWVpFYnhh

而且他还自制了5个常用大仓的差传点：

* 家具批发市场
  * https://socialclub.rockstargames.com/job/gtav/bVP0ivyoV0iSLJsjJMvT5Q
* 物流仓库
  * https://socialclub.rockstargames.com/job/gtav/gAiwYOjtQE2L2Idr0cGBeQ
* 达内尔兄弟仓库
  * https://socialclub.rockstargames.com/job/gtav/lFacb8aFy0q3lxKQ-qbjSQ
* 柏树仓库
  * https://socialclub.rockstargames.com/job/gtav/-0msxwdcJkavnSVnz-XRHg
* 西好麦坞
  * https://socialclub.rockstargames.com/job/gtav/QCayGLKZlEKfhfmYYkrgEQ

公益差传Bot也是他的功劳。这个我之后加入差传模块时，还会展开说明。

### 纯属娱乐il

[纯属娱乐il](https://space.bilibili.com/8578053) 开发了 `GTA_P quickly`：

* GTA循环自动取货小工具
  * https://www.bilibili.com/video/BV1mF54zMEaS

* 插件下载：
  * 链接: https://wwrc.lanzouu.com/b0pmngyzg
  * 密码: 6m51

这个项目对我实现大仓自动取货的模块帮助非常之大。

很多思路和细节，都是通过研究 `GTA_P quickly` 插件上的功能、提示和帮助，学习到的。
- 断网部分的各种技术细节，没有他在插件中的说明和帮助，我不知道会花费多少额外的时间
- 插件中自带的差传功能（用的是傲弗拉的差传bot），也非常好用

### MAGE安琪拉

[MAGE安琪拉](https://space.bilibili.com/2913798) 在 QuellGTA 中详细解释了断网原理：

* https://www.mageangela.cn/tools/QuellGTA-Notion.html

这里有几个实际演示的教程视频：

* 【GTAOL】QuellGTA实用功能演示（断网卡前置、085、地堡研究等）
  * https://www.bilibili.com/video/BV1Wc2jYgEkB

* 【GTAOL】防火墙法，读取罪神重置前的存档（罪神防重置 第二期）_游戏热门视频
  * https://www.bilibili.com/video/BV1VWPNeqEiN

### Dota涟漪

[Dota涟漪](https://space.bilibili.com/34236786) 分享了实际使用 `GTA_P quickly` 的详细过程，通过研究他的视频，我搞明白了几个关键点：

* GTA5 解答大仓取货遇到的问题
  * https://www.bilibili.com/video/BV1khCCBAEE4


## Developer Notes

See [Developer Notes](./DEV.md) for details of data collection, model training, runtime exporting, and real-time inference.