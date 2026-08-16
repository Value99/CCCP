"""Deprecated compatibility import for the public packed hybrid runtime.

New code must import :mod:`cccp.packed_hybrid`.  This module remains so older
Kimi deployments and third-party integrations do not break during the rename.
The implementation is model-independent and no longer lives here.
"""

from .packed_hybrid import *  # noqa: F401,F403
from .packed_hybrid import __all__
from .packed_hybrid import (  # legacy direct imports of internal helpers
    DeviceExpert,
    DevicePackedWeight,
    PackedExpert,
    _PackedArenas,
)
