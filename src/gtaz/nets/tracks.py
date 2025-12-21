"""GTA5 Enhanced 网络连接追踪器"""

"""
用于实时追踪 GTA5_Enhanced.exe 的网络连接状态变化。

安装依赖：
- psutil: 用于获取进程网络连接信息
- scapy: 用于更高级的网络数据包处理（可选）

如果要用 scapy，需要先安装 npcap：
- https://npcap.com/#download
- https://npcap.com/dist/npcap-1.85.exe

```sh
pip install psutil scapy
```
"""

import time
import socket
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from tclogger import TCLogger, logstr

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from scapy.all import sniff, IP, TCP, UDP, Raw
    from scapy.config import conf

    conf.verb = 0  # 禁用 scapy 详细输出
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


logger = TCLogger(name="NetTracker", use_prefix=True)


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

    # 流量统计
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    last_active: datetime = field(default_factory=datetime.now)

    # DNS 解析
    hostname: str = ""

    def __repr__(self) -> str:
        arrow = "->" if self.direction == "OUT" else "<-"
        # OUT 用 logstr.mesg (绿色), IN 用 logstr.file (蓝色)
        color_func = logstr.mesg if self.direction == "OUT" else logstr.file
        display = self.hostname if self.hostname else self.remote_addr
        remote_info = f"{display}:{self.remote_port}"
        return f"{arrow} {color_func(remote_info)}"

    def get_status_info(self) -> str:
        """获取状态信息字符串"""
        return f"{self.direction} {self.protocol} {self.status}"

    def get_traffic_info(self) -> str:
        """获取流量信息字符串"""
        sent = self._format_bytes(self.bytes_sent)
        recv = self._format_bytes(self.bytes_received)
        return f"↑{sent} ↓{recv} (↑{self.packets_sent}p ↓{self.packets_received}p)"

    @staticmethod
    def _format_bytes(bytes_count: int) -> str:
        """格式化字节数"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_count < 1024:
                return f"{bytes_count:.1f}{unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f}TB"


@dataclass
class PacketInfo:
    """数据包信息"""

    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    size: int
    direction: str  # "OUT" or "IN"
    timestamp: datetime = field(default_factory=datetime.now)
    payload_size: int = 0

    def __repr__(self) -> str:
        arrow = "->" if self.direction == "OUT" else "<-"
        return (
            f"[{self.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
            f"{self.direction} {self.protocol} {self.size:>5}B: "
            f"{self.src_ip}:{self.src_port} {arrow} {self.dst_ip}:{self.dst_port}"
        )


class GTAVNetworkTracker:
    """
    GTA5 Enhanced 网络连接追踪器。

    使用 psutil 追踪进程的网络连接状态变化，无需管理员权限。
    """

    def __init__(
        self,
        process_name: str = DEFAULT_PROCESS_NAME,
        on_connection: Optional[Callable[[NetworkConnection], None]] = None,
        on_packet: Optional[Callable[[PacketInfo], None]] = None,
        enable_packet_capture: bool = False,
    ):
        """
        初始化网络追踪器。

        :param process_name: 要追踪的进程名称
        :param on_connection: 连接变化时的回调函数
        :param on_packet: 捕获到数据包时的回调函数
        :param enable_packet_capture: 是否启用数据包捕获 (需要 scapy 和管理员权限)
        """
        self.process_name = process_name
        self.on_connection = on_connection
        self.on_packet = on_packet
        self.enable_packet_capture = enable_packet_capture and HAS_SCAPY

        self._running = False
        self._connection_thread: Optional[threading.Thread] = None
        self._packet_thread: Optional[threading.Thread] = None

        # 进程相关
        self._pid: Optional[int] = None
        self._process: Optional["psutil.Process"] = None

        # 连接追踪
        self._known_connections: set[tuple] = set()
        self._connection_map: dict[tuple, NetworkConnection] = {}  # 用于流量统计
        self._local_ports: set[int] = set()  # 用于数据包过滤

        # DNS 缓存
        self._dns_cache: dict[str, str] = {}

        # 统计信息
        self.stats = {
            "connections_total": 0,
            "connections_established": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "packets_sent": 0,
            "packets_received": 0,
        }
        self._start_time = datetime.now()

        # 检查依赖
        if not HAS_PSUTIL:
            logger.warn("psutil 未安装，连接追踪功能不可用")
            logger.hint("安装命令: pip install psutil")

        if self.enable_packet_capture and not HAS_SCAPY:
            logger.warn("scapy 未安装，数据包捕获功能不可用")
            logger.hint("安装命令: pip install scapy")
            logger.hint("还需要安装 Npcap: https://npcap.com/")
            self.enable_packet_capture = False

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

    def _resolve_hostname(self, ip: str) -> str:
        """
        解析 IP 地址为主机名（使用缓存）。

        :param ip: IP 地址
        :return: 主机名或原始 IP
        """
        if ip in self._dns_cache:
            return self._dns_cache[ip]

        try:
            hostname = socket.gethostbyaddr(ip)[0]
            self._dns_cache[ip] = hostname
            return hostname
        except (socket.herror, socket.gaierror, OSError):
            return ip

    def _handle_new_connection(self, conn: NetworkConnection):
        """
        处理新连接。

        :param conn: 网络连接对象
        """
        key = self._connection_key(conn)
        self._connection_map[key] = conn
        self._local_ports.add(conn.local_port)

        # 尝试 DNS 解析
        if conn.remote_addr and not conn.hostname:
            conn.hostname = self._resolve_hostname(conn.remote_addr)

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
        # 显示流量统计
        traffic_info = ""
        if key in self._connection_map:
            conn = self._connection_map.pop(key)
            if conn.bytes_sent > 0 or conn.bytes_received > 0:
                traffic_info = f" {conn.get_traffic_info()}"

        remote_addr, remote_port = key[2], key[3]
        protocol = key[4]
        direction = "OUT" if key[1] >= 1024 else "IN"
        arrow = "->" if direction == "OUT" else "<-"
        color_func = logstr.mesg if direction == "OUT" else logstr.file
        remote_info = f"{remote_addr}:{remote_port}"
        logger.warn(
            f"{arrow} {color_func(remote_info)} 连接关闭: {direction} {protocol}{traffic_info}"
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

    def _update_traffic_stats(self, src_port: int, dst_port: int, packet_size: int):
        """
        更新流量统计。

        :param src_port: 源端口
        :param dst_port: 目标端口
        :param packet_size: 数据包大小
        """
        # 查找匹配的连接
        for key, conn in self._connection_map.items():
            if src_port in self._local_ports:
                # 发送数据
                if key[1] == src_port:
                    conn.bytes_sent += packet_size
                    conn.packets_sent += 1
                    conn.last_active = datetime.now()
                    self.stats["bytes_sent"] += packet_size
                    self.stats["packets_sent"] += 1
                    break
            elif dst_port in self._local_ports:
                # 接收数据
                if key[1] == dst_port:
                    conn.bytes_received += packet_size
                    conn.packets_received += 1
                    conn.last_active = datetime.now()
                    self.stats["bytes_received"] += packet_size
                    self.stats["packets_received"] += 1
                    break

    def _packet_callback(self, packet):
        """
        数据包捕获回调。

        :param packet: scapy 数据包对象
        """
        if not IP in packet:
            return

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        # 确定协议和端口
        if TCP in packet:
            protocol = "TCP"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            protocol = "UDP"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return

        # 检查是否与我们的进程相关
        if src_port not in self._local_ports and dst_port not in self._local_ports:
            return

        # 确定方向
        direction = "OUT" if src_port in self._local_ports else "IN"

        # 更新流量统计
        packet_size = len(packet)
        self._update_traffic_stats(src_port, dst_port, packet_size)

        # 创建数据包信息
        packet_info = PacketInfo(
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            size=packet_size,
            direction=direction,
            payload_size=len(packet[Raw].load) if Raw in packet else 0,
        )

        if self.on_packet:
            self.on_packet(packet_info)

    def _capture_packets(self, interface: Optional[str] = None):
        """
        数据包捕获循环。

        :param interface: 网络接口名称，None 表示所有接口
        """
        if not HAS_SCAPY:
            logger.err("scapy 未安装，无法捕获数据包")
            return

        logger.note("开始数据包捕获...")
        logger.hint("提示: 需要管理员权限运行")

        try:
            # 只捕获 TCP 和 UDP 流量
            bpf_filter = "tcp or udp"

            sniff(
                iface=interface,
                filter=bpf_filter,
                prn=self._packet_callback,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except PermissionError:
            logger.err("权限不足，请以管理员身份运行")
        except Exception as e:
            logger.err(f"数据包捕获出错: {e}")

        logger.note("数据包捕获已停止")

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
        self._start_time = datetime.now()

        if HAS_PSUTIL:
            self._connection_thread = threading.Thread(
                target=self._track_connections, daemon=True
            )
            self._connection_thread.start()

        if self.enable_packet_capture and HAS_SCAPY:
            self._packet_thread = threading.Thread(
                target=self._capture_packets, daemon=True
            )
            self._packet_thread.start()
            logger.okay("网络追踪已启动（包含数据包捕获）")
        else:
            logger.okay("网络追踪已启动")

    def stop(self):
        """停止网络追踪。"""
        self._running = False

        if self._connection_thread:
            self._connection_thread.join(timeout=2.0)
            self._connection_thread = None

        if self._packet_thread:
            self._packet_thread.join(timeout=2.0)
            self._packet_thread = None

        logger.okay("网络追踪已停止")

    def log_stats(self):
        """打印统计信息。"""
        duration = (datetime.now() - self._start_time).total_seconds()

        logger.note("=== 网络统计 ===")
        logger.mesg(f"运行时长: {duration:.1f}秒")
        logger.mesg(f"总连接数: {self.stats['connections_total']}")
        logger.mesg(f"已建立连接: {self.stats['connections_established']}")

        # 流量统计
        sent = NetworkConnection._format_bytes(self.stats["bytes_sent"])
        recv = NetworkConnection._format_bytes(self.stats["bytes_received"])
        total = NetworkConnection._format_bytes(
            self.stats["bytes_sent"] + self.stats["bytes_received"]
        )

        logger.mesg(f"发送流量: ↑ {sent} ({self.stats['packets_sent']} 包)")
        logger.mesg(f"接收流量: ↓ {recv} ({self.stats['packets_received']} 包)")
        logger.mesg(f"总流量: {total}")

        # 速率统计
        if duration > 0:
            avg_rate_sent = self.stats["bytes_sent"] / duration
            avg_rate_recv = self.stats["bytes_received"] / duration
            avg_rate_total = (
                self.stats["bytes_sent"] + self.stats["bytes_received"]
            ) / duration

            logger.mesg(
                f"平均发送速率: {NetworkConnection._format_bytes(avg_rate_sent)}/s"
            )
            logger.mesg(
                f"平均接收速率: {NetworkConnection._format_bytes(avg_rate_recv)}/s"
            )
            logger.mesg(
                f"平均总速率: {NetworkConnection._format_bytes(avg_rate_total)}/s"
            )

        # 当前活跃连接
        active_count = len(self._connection_map)
        if active_count > 0:
            logger.note(f"\n当前活跃连接: {active_count}")
            for conn in list(self._connection_map.values())[:10]:  # 只显示前10个
                logger.mesg(
                    f"  {conn} {conn.get_status_info()} {conn.get_traffic_info()}"
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_tracker():
    """测试网络追踪器。"""
    # 默认启用数据包捕获以获取详细流量信息
    tracker = GTAVNetworkTracker(enable_packet_capture=True)

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
