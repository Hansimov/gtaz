# 自动取货模块 - 安装和使用教程

目前仅适配 GTAV 增强版。

## 安装 Python，创建 venv

本项目开发环境为 Python 3.13.9：
- https://www.python.org/downloads/release/python-3139/
- https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe

下载 `python-3.13.9-amd64.exe`，双击运行安装程序。

安装时注意勾选下面的 `Add Python 3.13 to PATH` 选项。

进入想放置虚拟环境的目录，创建 venv：

```sh
python -m venv gta
```

等待创建完成，激活：

```sh
call gta\Scripts\activate.bat
```

命令行左侧会显示 `(gta)`，说明已成功激活虚拟环境。

## 克隆项目，安装依赖

假设 repo 下载到：`D:/codes`。后续均以此路径作为参考。

克隆项目：

```sh
cd /d D:\codes
git clone --depth 1 https://github.com/Hansimov/gtaz.git 

# 如果速度慢，可以用镜像加速：
# git clone --depth 1 https://githubfast.com/Hansimov/gtaz.git 
```

如果项目未来有更新，运行下面的命令即可同步：

```sh
cd /d D:\codes\gtaz
git pull
```

安装依赖：

```sh
cd gtaz
pip install -r AUTO_PICKUP.txt -i https://mirrors.ustc.edu.cn/pypi/simple
```

过程中会出现 ViGEmBus 的安装提示（详见：[vgamepad](https://github.com/yannbouteiller/vgamepad?tab=readme-ov-file#windows)），确认并安装即可。

进入运行目录：

```sh
cd src
```

后续所有命令均在 `src` (`D:/codes/gtaz/src`) 目录下执行。

## 安装 VBCABLE，配置混音器

在自动取货模块中，需要根据下云的声音判断何时断网。

下载 VBCABLE：
- https://vb-audio.com/Cable/index.htm
- https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip

解压后，双击 `VBCABLE_Setup_x64.exe` 安装。

确保 GTAV 增强版正在运行。

在系统设置中，搜索 `混音器选项`，找到 GTAV 增强版：
- 输出（上面的选项）选择：`CABLE Input (VB-Audio Virtual Cable)`
- 输入（下面的选项）选择：`CABLE Output (VB-Audio Virtual Cable)`
- 注意：不是改系统声音输出，是在 `混音器选项` 里修改 GTAV 的输入和输出设备
- 注意：输出是 `Input`，输入是 `Output`，不要被字面意义误导

（这个小节的步骤将在未来的版本实现自动安装配置。）

## 测试和运行前检查清单

- [x] GTAV 增强版正在运行
  - 在 故事模式/在线模式/邀请战局 均可，只要不是在主菜单选择界面
  - 线下模式最好打到能自由活动的进度（把“难兄难弟”过了即可）
- [x] 优化图像设置：打开菜单 `设置` - `图像`
  - `屏幕类型` 改为 `窗口模式`，分辨率改为 `1024 x 768`
  - 可以不改，但是大部分开发和测试都是在这个分辨率下，改成这个会更稳定
- [x] 大仓员工已经派出取货
  - 正式运行前必须完成，测试时不需要
  - 一定要看到 `-$7500`，并且员工已经离开仓库
  - 哪几个仓库的员工派出了，就只给这几个仓库取货
- [x] 出生点设置为室内的地点
  - 推荐游戏厅（升级控制端后可以直接在游戏厅里查看大仓库存）
  - 不要选择机库，也避免有浴室的地点（比如公寓）
- [x] 混音器选项已正确配置
  - 输出为 `CABLE Input (VB-Audio Virtual Cable)`
  - 输入为 `CABLE Output (VB-Audio Virtual Cable)`
- [x] 命令行窗口已正确配置
  - 以管理员身份运行（因为后续需要添加和开关防火墙规则，用于存档断网）
  - 已激活 venv：左侧显示 `(gta)`
  - 已切换到 `D:/codes/gtaz/src` 目录

## 首次测试运行

添加防火墙规则：

```sh
python -m gtaz.nets.blocks -a
```

如果出现 `未能添加防火墙规则`，则需要以管理员身份运行命令行，然后再运行。

在员工取回货之前，运行下面的测试脚本，确认各个组件工作正常：

```sh
python -m gtaz.workers.auto_pickup -l -g
```

如果出现下面的情况，就说明运行正常：
- [x] 在下云的时候，命令行日志出现 `音频流已启动...匹配成功！...音频流已停止`，表明音频识别正常
- [x] 到线上后，左下角出现 `保存失败，Rockstar 云服务器现在不可用...`，表明断网成功

## 正式运行自动取货模块

确保退到线下故事模式：

```sh
python -m gtaz.workers.mode_switch -s
```

从最后一个员工外出取货开始计算，等待 48 分钟。

（保险起见可以等 50 分钟，后续版本会新增自动等待和开启。）

等待结束后，运行下面的脚本，开始自动取货（循环 85 次，基本能取满，可以按需调整）：

```sh
python -m gtaz.workers.auto_pickup -l -c 85
```

循环结束后，脚本会自动保存货物存档，并同步到云服务器。

<details> <summary><b>点击展开：其他实用脚本</b></summary>

## 其他实用脚本

### 模式和战局切换

从故事模式切换到线上邀请战局，或者从当前战局切换到新的邀请战局：

```sh
python -m gtaz.workers.mode_switch -i
```

从在线模式切换到故事模式：

```sh
python -m gtaz.workers.mode_switch -s
```

从故事模式切换到在线模式（公开战局）：

```sh
python -m gtaz.workers.mode_switch -o
```

### 防火墙规则管理

运行自动取货模块时，会自动 添加/启用/禁用 防火墙规则。不需要手动操作。

查看帮助：

```sh
python -m gtaz.nets.blocks -h
```

添加防火墙规则：

```sh
python -m gtaz.nets.blocks -a
```

删除防火墙规则：

```sh
python -m gtaz.nets.blocks -d
```

启用防火墙规则：

```sh
python -m gtaz.nets.blocks -e
```

禁用防火墙规则：

```sh
python -m gtaz.nets.blocks -s
```

</details>