"""pytest 共享夹具:仓库根加入 sys.path;注册表用临时 user 目录。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from launcher.profiles import ProfileRegistry  # noqa: E402
from launcher.resources import builtin_profile_dir  # noqa: E402


@pytest.fixture()
def registry(tmp_path):
    return ProfileRegistry(builtin_profile_dir(), tmp_path / "user")
