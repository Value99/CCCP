"""通用算子注册与能力选择。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .spec import OperatorCapability, OperatorRequest


OperatorImplementation = Callable[..., Any]


@dataclass(frozen=True)
class RegisteredOperator:
    name: str
    capability: OperatorCapability
    implementation: OperatorImplementation
    priority: int


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, RegisteredOperator] = {}
        self._resolved: dict[OperatorRequest, RegisteredOperator] = {}

    def register(
        self,
        name: str,
        capability: OperatorCapability,
        implementation: OperatorImplementation,
        *,
        priority: int = 0,
    ) -> None:
        """按稳定名称注册；重复导入同一后端是幂等的。"""
        normalized = OperatorCapability(
            operation=capability.operation.strip().lower(),
            device_types=tuple(
                value.strip().lower()
                for value in capability.device_types
            ),
            packed_formats=tuple(sorted(set(capability.packed_formats))),
            code_dims=tuple(sorted(set(capability.code_dims))),
            codebook_sizes=tuple(sorted(set(capability.codebook_sizes))),
            activations=tuple(
                value.strip().lower()
                for value in capability.activations
            ),
            max_top_k=int(capability.max_top_k),
            batch_sizes=tuple(sorted(set(capability.batch_sizes))),
            dtypes=tuple(sorted(set(
                value.strip().lower() for value in capability.dtypes
            ))),
            cache_formats=tuple(sorted(set(
                value.strip().lower()
                for value in capability.cache_formats
            ))),
            head_dims=tuple(sorted(set(
                int(value) for value in capability.head_dims
            ))),
            page_layouts=tuple(sorted(set(
                value.strip().lower()
                for value in capability.page_layouts
            ))),
            compression_ratios=tuple(sorted(set(
                int(value) for value in capability.compression_ratios
            ))),
            architecture_features=tuple(sorted(set(
                value.strip().lower()
                for value in capability.architecture_features
            ))),
        )
        item = RegisteredOperator(
            name=name,
            capability=normalized,
            implementation=implementation,
            priority=int(priority),
        )
        previous = self._operators.get(name)
        if previous is not None and previous != item:
            raise ValueError(f"operator {name!r} is already registered")
        self._operators[name] = item
        self._resolved.clear()

    def resolve(self, request: OperatorRequest) -> RegisteredOperator:
        request = request.normalized()
        cached = self._resolved.get(request)
        if cached is not None:
            return cached
        matches = [
            item
            for item in self._operators.values()
            if item.capability.supports(request)
        ]
        if not matches:
            raise LookupError(
                "没有匹配的算子实现："
                f"{request.normalized()!r}"
            )
        selected = max(
            matches,
            key=lambda item: (item.priority, item.name),
        )
        self._resolved[request] = selected
        return selected

    def call(self, request: OperatorRequest, **kwargs):
        return self.resolve(request).implementation(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))


REGISTRY = OperatorRegistry()
