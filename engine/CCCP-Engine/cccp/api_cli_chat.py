#!/usr/bin/env python3
"""通过 OpenAI 兼容接口与已启动的 CCCP 服务进行 CLI 多轮对话。"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request_json(url, *, method="GET", payload=None, api_key="", timeout=30):
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=timeout)


def discover_model(base_url, api_key, timeout):
    with request_json(f"{base_url}/models", api_key=api_key, timeout=timeout) as response:
        result = json.load(response)
    models = result.get("data") or []
    if not models:
        raise RuntimeError("服务没有返回可用模型")
    return models[0]["id"]


def chat(base_url, model, messages, api_key, stream, max_tokens, timeout):
    payload = {"model": model, "messages": messages, "stream": stream}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    with request_json(
        f"{base_url}/chat/completions",
        method="POST",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    ) as response:
        if not stream:
            result = json.load(response)
            message = result["choices"][0]["message"]
            return message.get("content") or message.get("reasoning_content") or ""

        pieces = []
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if text:
                print(text, end="", flush=True)
                pieces.append(text)
        print()
        return "".join(pieces)


def parse_args():
    parser = argparse.ArgumentParser(description="CCCP OpenAI 兼容服务 CLI 对话")
    parser.add_argument(
        "--base-url",
        default=os.getenv("CCCP_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="API 根地址（默认：http://127.0.0.1:8000/v1）",
    )
    parser.add_argument("--model", default=os.getenv("CCCP_MODEL"), help="模型名，默认自动获取")
    parser.add_argument("--api-key", default=os.getenv("CCCP_API_KEY", ""), help="可选 API Key")
    parser.add_argument("--system", default="", help="系统提示词")
    parser.add_argument("--max-tokens", type=int, default=None, help="每轮最多生成 token 数")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--timeout", type=float, default=600, help="单次请求超时秒数")
    parser.add_argument("--prompt", help="发送一条消息后退出，用于脚本或测试")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        model = args.model or discover_model(base_url, args.api_key, args.timeout)
    except Exception as exc:
        print(f"连接服务失败：{exc}", file=sys.stderr)
        return 1

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"已连接：{base_url}（模型：{model}）")
    if not args.prompt:
        print("命令：/clear 清空上下文，/exit 退出")

    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input("\n你> ").strip()
            if not prompt:
                if args.prompt is not None:
                    return 0
                continue
            if prompt in {"/exit", "/quit"}:
                return 0
            if prompt == "/clear":
                messages = messages[:1] if args.system else []
                print("上下文已清空")
                continue

            messages.append({"role": "user", "content": prompt})
            if args.prompt is None:
                print("助手> ", end="", flush=True)
            answer = chat(
                base_url,
                model,
                messages,
                args.api_key,
                not args.no_stream,
                args.max_tokens,
                args.timeout,
            )
            if args.no_stream:
                print(answer)
            messages.append({"role": "assistant", "content": answer})
            if args.prompt is not None:
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            return 0
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"\n请求失败（HTTP {exc.code}）：{detail}", file=sys.stderr)
            if args.prompt is not None:
                return 1
            messages.pop()
        except Exception as exc:
            print(f"\n请求失败：{exc}", file=sys.stderr)
            if args.prompt is not None:
                return 1
            messages.pop()


if __name__ == "__main__":
    raise SystemExit(main())
