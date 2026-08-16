"""End-user CCCP inference runtime entry point.

This entry point intentionally exposes only launcher-required inference and
route-scan commands.  Developer benchmarks and standalone API test clients are
not part of the offline end-user distribution.
"""

from __future__ import annotations

from importlib import import_module
import sys

from . import __version__


COMMANDS = ("launch", "check", "route-scan", "chat", "serve")
_COMMAND_DESCRIPTIONS = {
    "launch": "auto-detect model and start chat/API with production presets",
    "check": "validate model files, memory, GPUs and TP capacity",
    "route-scan": "scan a token corpus and create an expert residency profile",
    "chat": "interactive chat or one-shot generation",
    "serve": "OpenAI-compatible HTTP API server",
}


def _help_text() -> str:
    command_lines = "\n".join(
        f"  {command:<10} {_COMMAND_DESCRIPTIONS[command]}" for command in COMMANDS
    )
    return (
        "usage: python -m cccp <command> [options]\n\n"
        f"commands:\n{command_lines}\n\n"
        "Run `python -m cccp <command> --help` for command-specific help."
    )


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"-V", "--version"}:
        print(f"cccp-inference {__version__}")
        return
    if not argv or argv[0] in {"-h", "--help"}:
        print(_help_text())
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{_help_text()}")
        raise SystemExit(1)
    module = import_module(f".{command.replace('-', '_')}", __package__)
    module.main(rest)


if __name__ == "__main__":
    main()
