"""与模型名称无关的算子能力描述。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorRequest:
    """一次算子调用需要的能力。

    Kimi、GLM 和 DeepSeek 的差异只体现在这些字段及模型配置中，不进入
    注册键的实现名称。
    """

    operation: str
    device_type: str
    packed_formats: tuple[str, ...] = ()
    code_dims: tuple[int, ...] = ()
    codebook_sizes: tuple[int, ...] = ()
    activation: str = "none"
    top_k: int = 1
    batch_size: int = 1
    dtype: str = ""
    cache_format: str = ""
    head_dim: int = 0
    page_layout: str = ""
    compression_ratio: int = 0
    architecture_features: tuple[str, ...] = ()

    def normalized(self) -> "OperatorRequest":
        return OperatorRequest(
            operation=self.operation.strip().lower(),
            device_type=self.device_type.strip().lower(),
            packed_formats=tuple(sorted(set(self.packed_formats))),
            code_dims=tuple(sorted(set(int(v) for v in self.code_dims))),
            codebook_sizes=tuple(
                sorted(set(int(v) for v in self.codebook_sizes))
            ),
            activation=self.activation.strip().lower(),
            top_k=int(self.top_k),
            batch_size=int(self.batch_size),
            dtype=self.dtype.strip().lower(),
            cache_format=self.cache_format.strip().lower(),
            head_dim=int(self.head_dim),
            page_layout=self.page_layout.strip().lower(),
            compression_ratio=int(self.compression_ratio),
            architecture_features=tuple(sorted(set(
                value.strip().lower()
                for value in self.architecture_features
            ))),
        )


@dataclass(frozen=True)
class OperatorCapability:
    operation: str
    device_types: tuple[str, ...]
    packed_formats: tuple[str, ...] = ()
    code_dims: tuple[int, ...] = ()
    codebook_sizes: tuple[int, ...] = ()
    activations: tuple[str, ...] = ()
    max_top_k: int = 1
    batch_sizes: tuple[int, ...] = (1,)
    dtypes: tuple[str, ...] = ()
    cache_formats: tuple[str, ...] = ()
    head_dims: tuple[int, ...] = ()
    page_layouts: tuple[str, ...] = ()
    compression_ratios: tuple[int, ...] = ()
    architecture_features: tuple[str, ...] = ()

    def supports(self, request: OperatorRequest) -> bool:
        request = request.normalized()
        return (
            request.operation == self.operation
            and request.device_type in self.device_types
            and set(request.packed_formats).issubset(self.packed_formats)
            and set(request.code_dims).issubset(self.code_dims)
            and set(request.codebook_sizes).issubset(self.codebook_sizes)
            and request.activation in self.activations
            and 0 < request.top_k <= self.max_top_k
            and request.batch_size in self.batch_sizes
            and (
                not request.dtype
                or request.dtype in self.dtypes
            )
            and (
                not request.cache_format
                or request.cache_format in self.cache_formats
            )
            and (
                request.head_dim <= 0
                or request.head_dim in self.head_dims
            )
            and (
                not request.page_layout
                or request.page_layout in self.page_layouts
            )
            and (
                request.compression_ratio <= 0
                or request.compression_ratio in self.compression_ratios
            )
            and set(request.architecture_features).issubset(
                self.architecture_features
            )
        )

