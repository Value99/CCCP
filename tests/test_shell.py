"""Desktop shell must stay hidden until the first styled WebView frame is painted."""
from types import SimpleNamespace

import launcher.shell as shell


def test_window_is_hidden_until_loaded(monkeypatch):
    calls = {"shown": 0, "stopped": 0, "joined": None}

    class LoadedEvent:
        def __init__(self):
            self.callback = None

        def __iadd__(self, callback):
            self.callback = callback
            return self

    loaded = LoadedEvent()
    window = SimpleNamespace(
        events=SimpleNamespace(loaded=loaded),
        show=lambda: calls.__setitem__("shown", calls["shown"] + 1),
    )

    class FakeWebview:
        @staticmethod
        def create_window(*args, **kwargs):
            calls["window_kwargs"] = kwargs
            return window

        @staticmethod
        def start(**kwargs):
            assert calls["shown"] == 0
            loaded.callback()
            assert calls["shown"] == 0
            calls["window_kwargs"]["js_api"].win_ready()
            calls["window_kwargs"]["js_api"].win_ready()

    server = SimpleNamespace(started=True, should_exit=False)
    thread = SimpleNamespace(
        is_alive=lambda: False,
        join=lambda timeout=None: calls.__setitem__("joined", timeout),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            adapter=SimpleNamespace(
                stop=lambda: calls.__setitem__("stopped", calls["stopped"] + 1)
            )
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", FakeWebview)
    monkeypatch.setattr(shell, "_start_uvicorn", lambda *args: (server, thread))

    shell.run_with_shell(app, "127.0.0.1", 8790)

    assert calls["window_kwargs"]["hidden"] is True
    assert calls["window_kwargs"]["background_color"] == shell.BG_COLOR
    assert calls["shown"] == 1
    assert calls["stopped"] == 1
    assert calls["joined"] == 20.0
    assert server.should_exit is True


def test_native_shell_failure_never_opens_browser(monkeypatch):
    calls = {"message": "", "joined": None}

    class BrokenWebview:
        @staticmethod
        def create_window(*args, **kwargs):
            raise RuntimeError("WebView2 unavailable")

    server = SimpleNamespace(started=True, should_exit=False)

    class FakeThread:
        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def join(timeout=None):
            calls["joined"] = timeout

    monkeypatch.setitem(__import__("sys").modules, "webview", BrokenWebview)
    monkeypatch.setattr(shell, "_start_uvicorn", lambda *args: (server, FakeThread()))
    monkeypatch.setattr(shell, "_show_shell_error", lambda message: calls.__setitem__("message", message))

    shell.run_with_shell(object(), "127.0.0.1", 8790)

    assert server.should_exit is True
    assert calls["joined"] == 20.0
    assert "WebView2 unavailable" in calls["message"]
