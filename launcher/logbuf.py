"""内存环形日志缓冲:「终端」页可直接读取启动器(含 uvicorn)日志。"""
from __future__ import annotations

import logging
from collections import deque

_CAPACITY = 500
_records: deque[str] = deque(maxlen=_CAPACITY)
_attached = False

# 前端轮询类 access 噪音过滤(只滤轮询行,保留真实业务日志)
_NOISE = ('"/api/terminal', '"GET /api/health', '"GET /api/launch/status')


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        if any(n in line for n in _NOISE):
            return
        _records.append(line)


def attach_ring_log() -> None:
    """挂到 root logger(幂等)。uvicorn 各 logger 默认 propagate 到 root。"""
    global _attached
    if _attached:
        return
    h = _RingHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(h)
    _attached = True
    logging.getLogger("winui").info("环形日志缓冲已挂载(capacity=%d)", _CAPACITY)


def tail_lines(n: int = 300) -> list[str]:
    lines = list(_records)
    return lines[-n:] if n else lines
