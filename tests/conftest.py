"""pytest 共享夹具：模型配置来自模型目录 profiles/。"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from launcher.profiles import ProfileRegistry  # noqa: E402

MODEL = Path(
    os.environ.get(
        "CCCP_TEST_DSV4_MODEL",
        str(ROOT / "models" / "dsv4-cccp-s-noblack-v2"),
    )
)


@pytest.fixture()
def registry(tmp_path):
    if not (MODEL / "cccp.json").is_file():
        pytest.skip("本机未提供集成测试模型")
    return ProfileRegistry([MODEL], tmp_path / "user")
