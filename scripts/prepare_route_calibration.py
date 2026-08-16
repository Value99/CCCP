#!/usr/bin/env python3
"""从角色扮演语料确定性抽取小型路由校准提示，不输出提示正文。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CATEGORIES = {
    "roleplay-romance": ["爱", "恋", "喜欢", "亲吻", "拥抱", "恋人", "男友", "女友"],
    "roleplay-immersive": ["角色", "动作", "场景", "眼神", "声音", "扮演"],
    "roleplay-worldbuilding": ["世界", "背景", "剧情", "故事", "王国", "城市"],
}


def clean(content: object) -> str:
    text = str(content or "").replace("\r", " ").replace("\n", " ").strip()
    if "<|channel>thought" in text and "<channel|>" in text:
        text = text.split("<channel|>", 1)[1].strip()
    return " ".join(text.split())


def choose(candidates: list[tuple[str, str, int]], seed: int) -> tuple[str, str, int]:
    if not candidates:
        raise RuntimeError("语料中没有找到该类别候选")
    return min(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[2]}:{item[0]}".encode("utf-8")
        ).digest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--max-chars", type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.max_chars <= 512:
        raise SystemExit("max-chars 必须在 2..512")

    candidates = {category: [] for category in CATEGORIES}
    valid_lines = invalid_lines = 0
    with args.corpus.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            messages = item.get("messages") if isinstance(item, dict) else None
            if not isinstance(messages, list):
                invalid_lines += 1
                continue
            valid_lines += 1
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                    continue
                text = clean(message.get("content"))
                for category, keywords in CATEGORIES.items():
                    for keyword in keywords:
                        position = text.find(keyword)
                        if position < 0:
                            continue
                        half = max(1, (args.max_chars - len(keyword)) // 2)
                        start = max(0, position - half)
                        snippet = text[start:start + args.max_chars]
                        if snippet:
                            candidates[category].append((snippet, keyword, line_no))
                        break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "format": "winui-route-calibration-prompts-v1",
        "corpus_name": args.corpus.name,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "valid_lines": valid_lines,
        "invalid_lines": invalid_lines,
        "seed": args.seed,
        "max_chars": args.max_chars,
        "privacy": "提示正文仅写入 data/runtime 临时文件；审计只保存哈希和长度",
        "categories": {},
    }
    for category, items in candidates.items():
        snippet, keyword, line_no = choose(items, args.seed)
        path = args.output_dir / f"{category}.prompt.txt"
        path.write_text(snippet, encoding="utf-8")
        audit["categories"][category] = {
            "prompt_file": path.name,
            "prompt_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
            "characters": len(snippet),
            "keyword_sha256": hashlib.sha256(keyword.encode("utf-8")).hexdigest(),
            "source_line_sha256": hashlib.sha256(str(line_no).encode("ascii")).hexdigest(),
            "candidates": len(items),
        }
    audit_path = args.output_dir / "route-calibration-audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
