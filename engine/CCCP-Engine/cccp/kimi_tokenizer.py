"""Small dependency-light tokenizer adapter for Kimi K3.

The model archive carries a ``tiktoken.model`` rather than Hugging Face's
``tokenizer.json``.  Only the ``tiktoken`` package is required here; loading
the remote model code, Transformers and the Rust ``tokenizers`` package is
unnecessary for CCCP inference.
"""

from __future__ import annotations

import codecs
import json
import os
from dataclasses import dataclass
from pathlib import Path


_PATTERN = "|".join([
    r"[\p{Han}]+",
    (
        r"[^\r\n\p{L}\p{N}]?"
        r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    ),
    (
        r"[^\r\n\p{L}\p{N}]?"
        r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    ),
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
])


@dataclass(frozen=True)
class KimiEncoding:
    ids: list[int]


class KimiDecodeStream:
    """Incremental UTF-8 decoder compatible with chat_service's stream API."""

    def __init__(self, *, skip_special_tokens: bool = False):
        self.skip_special_tokens = bool(skip_special_tokens)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def step(self, tokenizer: "KimiTokenizer", token_id: int) -> str:
        token_id = int(token_id)
        if self.skip_special_tokens and token_id in tokenizer.special_ids:
            return ""
        return self._decoder.decode(
            tokenizer.decode_single_token_bytes(token_id),
            final=False,
        )


class KimiTokenizer:
    """Engine-compatible wrapper around Kimi's published tiktoken format."""

    num_reserved_special_tokens = 256

    def __init__(self, model_dir: str):
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
        except ImportError as exc:
            raise RuntimeError(
                "Kimi tokenizer requires the lightweight 'tiktoken' package"
            ) from exc

        vocab_file = os.path.join(model_dir, "tiktoken.model")
        config_file = os.path.join(model_dir, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        explicit: dict[int, str] = {}
        for key, value in config.get("added_tokens_decoder", {}).items():
            explicit[int(key)] = str(value["content"])

        mergeable_ranks = load_tiktoken_bpe(vocab_file)
        base = len(mergeable_ranks)
        self.special_tokens = {
            explicit.get(token_id, f"<|reserved_token_{token_id}|>"):
            token_id
            for token_id in range(
                base,
                base + self.num_reserved_special_tokens,
            )
        }
        self.special_ids = frozenset(self.special_tokens.values())
        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name,
            pat_str=_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    @staticmethod
    def _split_long_runs(text: str, limit: int = 25_000):
        if not text:
            return
        start = 0
        run = 0
        is_space = text[0].isspace()
        for index, char in enumerate(text):
            current = char.isspace()
            if current != is_space:
                run = 1
                is_space = current
            else:
                run += 1
                if run > limit:
                    yield text[start:index]
                    start = index
                    run = 1
        yield text[start:]

    def encode(
        self,
        text: str,
        *,
        allow_special_tokens: bool = True,
    ) -> KimiEncoding:
        ids: list[int] = []
        for outer in range(0, len(text), 400_000):
            piece = text[outer:outer + 400_000]
            for chunk in self._split_long_runs(piece):
                if allow_special_tokens:
                    ids.extend(self.model.encode(
                        chunk,
                        allowed_special="all",
                    ))
                else:
                    ids.extend(self.model.encode(
                        chunk,
                        disallowed_special=(),
                    ))
        return KimiEncoding(ids)

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        values = [int(value) for value in ids]
        if skip_special_tokens:
            values = [
                value for value in values
                if value not in self.special_ids
            ]
        return self.model.decode(values)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.model.decode_single_token_bytes(int(token_id))

    def id_to_token(self, token_id: int) -> str | None:
        token_id = int(token_id)
        for text, value in self.special_tokens.items():
            if value == token_id:
                return text
        try:
            return self.decode_single_token_bytes(token_id).decode(
                "utf-8",
                errors="replace",
            )
        except KeyError:
            return None

    def new_decode_stream(
        self,
        *,
        skip_special_tokens: bool = False,
    ) -> KimiDecodeStream:
        return KimiDecodeStream(
            skip_special_tokens=skip_special_tokens,
        )


__all__ = [
    "KimiDecodeStream",
    "KimiEncoding",
    "KimiTokenizer",
]
