@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo  GTAV 自动取货工具 - 打包脚本
echo ============================================================
echo.

:: 检查 Python 是否安装
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保已安装 Python 并添加到 PATH
    pause
    exit /b 1
)

:: 检查 PyInstaller 是否安装
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [提示] 正在安装 PyInstaller...
    pip install pyinstaller -i https://mirrors.ustc.edu.cn/pypi/simple
    if errorlevel 1 (
        echo [错误] PyInstaller 安装失败
        pause
        exit /b 1
    )
)

:: 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: 创建输出目录
if not exist "dist" mkdir dist

echo.
echo [1/4] 清理旧的构建文件...
if exist "build" rmdir /s /q "build"
if exist "dist\pickup.exe" del /f /q "dist\pickup.exe"
if exist "dist\switch.exe" del /f /q "dist\switch.exe"
if exist "dist\blocks.exe" del /f /q "dist\blocks.exe"

echo.
echo [2/4] 检查依赖是否已安装...
pip show acto tclogger vgamepad numpy opencv-python Pillow psutil sounddevice soundfile >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装项目依赖...
    pip install -e .[pickup] -i https://mirrors.ustc.edu.cn/pypi/simple
)

echo.
echo [3/4] 开始打包...
echo.

:: 使用 spec 文件打包
pyinstaller gtaz.spec --noconfirm

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo [4/4] 打包完成！
echo.
echo 生成的文件位于 dist 目录：
if exist "dist\pickup.exe" (
    echo   - pickup.exe  ^(自动取货主程序^)
    for %%A in ("dist\pickup.exe") do echo     文件大小: %%~zA bytes
)
if exist "dist\switch.exe" (
    echo   - switch.exe  ^(模式切换工具^)
    for %%A in ("dist\switch.exe") do echo     文件大小: %%~zA bytes
)
if exist "dist\blocks.exe" (
    echo   - blocks.exe  ^(防火墙规则管理^)
    for %%A in ("dist\blocks.exe") do echo     文件大小: %%~zA bytes
)

echo.
echo ============================================================
echo  使用说明：
echo ============================================================
echo.
echo  1. 将 dist 目录下的 exe 文件复制到任意位置
echo  2. 以管理员身份运行命令提示符
echo  3. 使用方式与 Python 脚本相同：
echo.
echo     pickup -l              # 循环1次（测试）
echo     pickup -l -w 48 -c 85  # 等待48分钟后开始，循环85次
echo     switch -s              # 切换到故事模式
echo     switch -i              # 切换到邀请战局
echo     blocks -a              # 添加防火墙规则
echo     blocks -e              # 启用防火墙规则
echo.
echo  注意：首次运行前请确保已安装 VBCABLE 和 ViGEmBus
echo ============================================================
echo.

pause
