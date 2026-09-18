import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str
    api_endpoint: str
    model_name: str
    api_key: str
    rate_limit: int = 10  # 每分钟请求限制
    max_workers: int = 5  # 最大并发数
    extra_headers: Optional[Dict[str, str]] = field(default_factory=dict)  # 模型特有请求头，如 version

class ThreadSafeCounter:
    """线程安全计数器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value