"""原生桌面窗口外壳(pywebview/WebView2)。

- 后端 uvicorn 跑在后台线程;窗口加载本机 HTTP 地址。
- frameless=True:去掉系统标题栏,由前端自绘(标题栏按钮通过 js_api 调用)。
- pywebview 或 WebView2 运行时不可用时明确报错并结束，不启动外部浏览器。
- 冻结(PyInstaller)与开发环境同一入口:run_with_shell(app, host, port)。
"""
from __future__ import annotations

import logging
import threading
import time
import ctypes

import uvicorn

log = logging.getLogger("winui.shell")

WINDOW_TITLE = "CCCP 启动器 — CCCP-Engine"
BG_COLOR = "#eee3cd"  # 与暖色主题浅色底一致，避免原生窗口首帧闪白


class _WinCtlApi:
    """无边框窗口的窗口控制 JS API(标题栏按钮调用)。"""

    def __init__(self) -> None:
        self._window = None
        self._maxed = False
        self._shown = False
        self._show_lock = threading.Lock()

    def bind(self, window) -> None:
        self._window = window

    def win_ready(self) -> None:
        """CSS 完成首帧绘制后再显示窗口，避免 WebView2 白色过渡帧。"""
        with self._show_lock:
            if self._window and not self._shown:
                self._window.show()
                self._shown = True

    def win_minimize(self) -> None:
        if self._window:
            self._window.minimize()

    def win_toggle_max(self) -> None:
        w = self._window
        if not w:
            return
        maximize = getattr(w, "maximize", None)
        restore = getattr(w, "restore", None)
        if callable(maximize) and callable(restore):
            self._maxed = not self._maxed
            (maximize if self._maxed else restore)()
        elif hasattr(w, "toggle_fullscreen"):
            w.toggle_fullscreen()

    def win_close(self) -> None:
        if self._window:
            self._window.destroy()

    def win_move(self, x: float, y: float) -> None:
        """前端标题栏拖拽:把窗口移到屏幕坐标(x, y)。"""
        if self._window:
            self._window.move(int(x), int(y))


def _start_uvicorn(app, host: str, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    # PyInstaller windowed EXE has no stdout/stderr.  Explicitly disabling
    # ANSI colour detection prevents uvicorn's formatter from calling
    # ``sys.stdout.isatty()`` on None during desktop startup.
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", use_colors=False
    )
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


def _show_shell_error(message: str) -> None:
    """在 windowed EXE 没有控制台时仍给出可见的原生错误。"""
    if not hasattr(ctypes, "windll"):
        return
    try:
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "CCCP 启动器",
            0x10,  # MB_ICONERROR | MB_OK
        )
    except (AttributeError, OSError):
        pass


def run_with_shell(app, host: str, port: int) -> None:
    """启动后端 + 原生无边框窗口；桌面外壳失败时安全退出。"""
    url = f"http://{host}:{port}"
    server, thread = _start_uvicorn(app, host, port)
    shell_error: Exception | None = None

    try:
        import webview  # pywebview,依赖 WebView2 运行时

        api = _WinCtlApi()
        window = webview.create_window(
            WINDOW_TITLE, url,
            width=1440, height=900, min_size=(1024, 640),
            background_color=BG_COLOR,
            hidden=True,  # 等 HTML/CSS 首帧完成再显示，避免 WebView2 默认白底闪屏
            frameless=True,
            easy_drag=False,  # 关闭全局拖拽(否则吞掉正文文本选择);拖拽仅挂自绘标题栏
            text_select=True,
            js_api=api,
        )
        api.bind(window)
        # 正常路径由页面在连续两帧绘制后调用 win_ready。若页面脚本损坏，
        # 仍在 3 秒后显示有主题底色的窗口，避免永远隐藏。
        def loaded_fallback() -> None:
            timer = threading.Timer(3.0, api.win_ready)
            timer.daemon = True
            timer.start()

        window.events.loaded += loaded_fallback
        webview.start(debug=False)
    except Exception as exc:  # noqa: BLE001 —— GUI 错误必须可见且不能残留后端
        shell_error = exc
    finally:
        # webview.start() 会在最后一个原生窗口关闭后返回。必须先停止推理
        # 子进程，再等待 uvicorn 完成 FastAPI shutdown；仅设置 should_exit
        # 后立刻让冻结主进程退出会把 python.exe 模型进程遗留在后台。
        adapter = getattr(getattr(app, "state", None), "adapter", None)
        stop = getattr(adapter, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001 —— 继续完成服务线程回收
                log.exception("关闭 CCCP 推理子进程失败")
        server.should_exit = True
        join = getattr(thread, "join", None)
        if callable(join):
            join(timeout=20.0)

    if shell_error is not None:
        log.exception(
            "原生窗口不可用(%s): %s",
            type(shell_error).__name__,
            url,
            exc_info=(
                type(shell_error),
                shell_error,
                shell_error.__traceback__,
            ),
        )
        _show_shell_error(
            "无法创建 CCCP 原生窗口。\n\n"
            "请确认系统 WebView2 运行时完整，或重新安装离线启动器。\n"
            f"错误：{type(shell_error).__name__}: {shell_error}"
        )
