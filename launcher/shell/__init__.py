"""原生桌面窗口外壳(pywebview/WebView2)。

- 后端 uvicorn 跑在后台线程;窗口加载本机 HTTP 地址。
- pywebview 或 WebView2 运行时不可用时,自动降级为"系统默认浏览器打开"。
- 冻结(PyInstaller)与开发环境同一入口:run_with_shell(app, host, port)。
"""
from __future__ import annotations

import logging
import threading
import time
import webbrowser

import uvicorn

log = logging.getLogger("winui.shell")

WINDOW_TITLE = "WINUI-EXE — TPQ-Final 启动器"
BG_COLOR = "#0b0e14"  # 与前端深色系一致,避免白屏闪烁


def _start_uvicorn(app, host: str, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    return server, thread


def _wait_forever(thread: threading.Thread, server: uvicorn.Server) -> None:
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True


def run_with_shell(app, host: str, port: int) -> None:
    """启动后端 + 原生窗口;失败降级为浏览器模式(不阻塞应用交付)。"""
    url = f"http://{host}:{port}"
    server, thread = _start_uvicorn(app, host, port)

    try:
        import webview  # pywebview,依赖 WebView2 运行时

        window = webview.create_window(
            WINDOW_TITLE, url,
            width=1440, height=900, min_size=(1024, 640),
            background_color=BG_COLOR,
            text_select=True,
        )
        webview.start(debug=False)
        server.should_exit = True  # 窗口关闭 -> 关停后端
        return
    except Exception as exc:  # noqa: BLE001 —— 任何 GUI 失败都降级
        log.warning("原生窗口不可用(%s),降级为浏览器模式: %s", type(exc).__name__, url)

    webbrowser.open(url)
    _wait_forever(thread, server)
