"""GTA5 Enhanced 网络连接追踪器"""

"""
用于实时追踪 GTA5_Enhanced.exe 的网络连接状态变化。

Requirements:
- psutil: 用于获取进程网络连接信息

安装依赖:
    pip install psutil
"""

import time
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from tclogger import TCLogger, logstr

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


logger = TCLogger(name="GTAVNetTracker", use_prefix=True)


# 默认进程名
DEFAULT_PROCESS_NAME = "GTA5_Enhanced.exe"


@dataclass
class NetworkConnection:
    """网络连接信息"""

    local_addr: str
    local_port: int
    remote_addr: str
    remote_port: int
    status: str
    protocol: str = "TCP"
    direction: str = "OUT"  # "OUT" 表示发送(主动连接), "IN" 表示接收(被动连接)
    timestamp: datetime = field(default_factory=datetime.now)
    pid: int = 0
    process_name: str = ""

    def __repr__(self) -> str:
        arrow = "->" if self.direction == "OUT" else "<-"
        # OUT 用 logstr.mesg (绿色), IN 用 logstr.file (蓝色)
        color_func = logstr.mesg if self.direction == "OUT" else logstr.file
        remote_info = f"{self.remote_addr}:{self.remote_port}"
        return f"{arrow} {color_func(remote_info)}"

    def get_status_info(self) -> str:
        """获取状态信息字符串"""
        return f"{self.direction} {self.protocol} {self.status}"


class GTAVNetworkTracker:
    """
    GTA5 Enhanced 网络连接追踪器。

    使用 psutil 追踪进程的网络连接状态变化，无需管理员权限。
    """

    def __init__(
        self,
        process_name: str = DEFAULT_PROCESS_NAME,
        on_connection: Optional[Callable[[NetworkConnection], None]] = None,
    ):
        """
        初始化网络追踪器。

        :param process_name: 要追踪的进程名称
        :param on_connection: 连接变化时的回调函数
        """
        self.process_name = process_name
        self.on_connection = on_connection

        self._running = False
        self._connection_thread: Optional[threading.Thread] = None

        # 进程相关
        self._pid: Optional[int] = None
        self._process: Optional["psutil.Process"] = None

        # 连接追踪
        self._known_connections: set[tuple] = set()

        # 统计信息
        self.stats = {
            "connections_total": 0,
            "connections_established": 0,
        }

        # 检查依赖
        if not HAS_PSUTIL:
            logger.warn("psutil 未安装，连接追踪功能不可用")
            logger.hint("安装命令: pip install psutil")

    def find_process(self) -> Optional[int]:
        """
        查找目标进程。

        :return: 进程 PID，未找到则返回 None
        """
        if not HAS_PSUTIL:
            return None

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == self.process_name:
                    self._pid = proc.info["pid"]
                    self._process = psutil.Process(self._pid)
                    logger.okay(f"找到进程: {self.process_name} (PID: {self._pid})")
                    return self._pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        logger.warn(f"未找到进程: {self.process_name}")
        return None

    def _determine_direction(self, conn) -> str:
        """
        判断连接方向。

        :param conn: psutil 连接对象
        :return: "OUT" 或 "IN"
        """
        if conn.status == "LISTEN":
            return "IN"

        if conn.status == "ESTABLISHED" and conn.raddr:
            # 临时端口范围通常表示主动连接
            if conn.laddr.port >= 49152 or (
                conn.laddr.port > 1024 and conn.raddr.port <= 1024
            ):
                return "OUT"
            return "IN"

        return "OUT"

    def _create_connection(self, conn) -> NetworkConnection:
        """
        从 psutil 连接对象创建 NetworkConnection。

        :param conn: psutil 连接对象
        :return: NetworkConnection 实例
        """
        return NetworkConnection(
            local_addr=conn.laddr.ip,
            local_port=conn.laddr.port,
            remote_addr=conn.raddr.ip if conn.raddr else "",
            remote_port=conn.raddr.port if conn.raddr else 0,
            status=conn.status,
            protocol="TCP" if conn.type == socket.SOCK_STREAM else "UDP",
            direction=self._determine_direction(conn),
            pid=self._pid or 0,
            process_name=self.process_name,
        )

    def get_connections(self) -> list[NetworkConnection]:
        """
        获取进程当前的所有网络连接。

        :return: 网络连接列表
        """
        if not HAS_PSUTIL or not self._process:
            return []

        connections = []
        try:
            for conn in self._process.net_connections(kind="inet"):
                if conn.laddr:
                    connections.append(self._create_connection(conn))
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warn(f"获取连接失败: {e}")

        return connections

    def _connection_key(self, conn: NetworkConnection) -> tuple:
        """生成连接的唯一标识键。"""
        return (
            conn.local_addr,
            conn.local_port,
            conn.remote_addr,
            conn.remote_port,
            conn.protocol,
        )

    def _handle_new_connection(self, conn: NetworkConnection):
        """
        处理新连接。

        :param conn: 网络连接对象
        """
        self.stats["connections_total"] += 1
        if conn.status == "ESTABLISHED":
            self.stats["connections_established"] += 1

        logger.okay(f"{conn} 连接创建: {conn.get_status_info()}")
        if self.on_connection:
            self.on_connection(conn)

    def _handle_closed_connection(self, key: tuple):
        """
        处理关闭的连接。

        :param key: 连接的唯一标识键 (local_addr, local_port, remote_addr, remote_port, protocol)
        """
        remote_addr, remote_port = key[2], key[3]
        protocol = key[4]
        direction = "OUT" if key[1] >= 1024 else "IN"
        arrow = "->" if direction == "OUT" else "<-"
        color_func = logstr.mesg if direction == "OUT" else logstr.file
        remote_info = f"{remote_addr}:{remote_port}"
        logger.warn(
            f"{arrow} {color_func(remote_info)} 连接关闭: {direction} {protocol}"
        )

    def _track_connections(self, interval: float = 0.5):
        """
        连接追踪循环。

        :param interval: 检查间隔 (秒)
        """
        logger.note("开始连接追踪...")

        while self._running:
            if not self._process or not self._process.is_running():
                # 进程不存在，尝试重新查找
                if not self.find_process():
                    time.sleep(interval * 2)
                    continue

            try:
                current_connections = self.get_connections()
                current_keys = set()

                # 处理当前连接
                for conn in current_connections:
                    key = self._connection_key(conn)
                    current_keys.add(key)

                    if key not in self._known_connections:
                        self._known_connections.add(key)
                        self._handle_new_connection(conn)

                # 处理关闭的连接
                closed_keys = self._known_connections - current_keys
                for key in closed_keys:
                    self._known_connections.discard(key)
                    self._handle_closed_connection(key)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None
                self._pid = None

            time.sleep(interval)

        logger.note("连接追踪已停止")

    def start(self):
        """
        启动网络追踪。
        """
        if self._running:
            logger.warn("追踪器已在运行")
            return

        # 查找进程
        if not self.find_process():
            logger.warn("未找到目标进程，将在后台持续尝试...")

        self._running = True

        if HAS_PSUTIL:
            self._connection_thread = threading.Thread(
                target=self._track_connections, daemon=True
            )
            self._connection_thread.start()

        logger.okay("网络追踪已启动")

    def stop(self):
        """停止网络追踪。"""
        self._running = False

        if self._connection_thread:
            self._connection_thread.join(timeout=2.0)
            self._connection_thread = None

        logger.okay("网络追踪已停止")

    def log_stats(self):
        """打印统计信息。"""
        logger.note("=== 网络统计 ===")
        logger.mesg(f"总连接数: {self.stats['connections_total']}")
        logger.mesg(f"已建立连接: {self.stats['connections_established']}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_tracker():
    """测试网络追踪器。"""
    tracker = GTAVNetworkTracker()

    try:
        tracker.start()

        logger.hint("按 Ctrl+C 停止追踪")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.note("\n正在停止...")
    finally:
        tracker.stop()
        tracker.log_stats()


if __name__ == "__main__":
    test_tracker()

    # python -m gtaz.nets.tracks
