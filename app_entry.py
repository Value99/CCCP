"""PyInstaller 冻结入口。

直接以 `launcher/app.py` 作入口会让包内相对导入失败(from . import …),
所以冻结版从包外导入再调用 main()。开发环境仍可用 `python -m launcher.app`。
"""
from launcher.app import main

if __name__ == "__main__":
    main()
