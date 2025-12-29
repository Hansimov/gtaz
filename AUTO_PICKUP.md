# 自动取货模块 - 安装和使用教程

目前仅适配 GTAV 增强版。

## 安装 Python，创建 venv

本项目开发环境为 Python 3.13.9：
- https://www.python.org/downloads/release/python-3139/
- https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe

创建 venv：

```sh
# cd <path_you_like>
python -m venv gta
```

激活 venv：

```sh
# cd <path_you_like>
call gta\Scripts\activate.bat
```

## 安装 VB-Audio，配置混音器

在自动取货模块中，需要根据下云的声音判断何时断网。

下载 VB-Audio：
- https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip

解压后，双击 `VBCABLE_Setup_x64.exe` 安装。

确保 GTA5 增强版正在运行。

在系统设置中，搜索 `混音器选项`，找到 GTAV 增强版：
- 输出（上面的选项）选择：`CABLE Input (VB-Audio Virtual Cable)`
- 输入（下面的选项）选择：`CABLE Output (VB-Audio Virtual Cable)`
- 注意：输出是 `CABLE Input`，输入是 `CABLE Output`

## 安装依赖