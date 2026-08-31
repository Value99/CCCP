"""OpenAI-compatible HTTP schemas and routes for CCCP chat services.

The application factory is deliberately dependency-injected.  Model loading
belongs to the server entry point, not to this transport module.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import hmac
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from .chat_adapters import (
    ChatMessage,
    ChatOptions,
    StreamDelta,
    ToolCall,
    ToolFunction,
    UnsupportedChatCapability,
)
from .chat_adapters.base import ChatContentPart
from .chat_service import ChatQueueFull, ChatService, GenerationReady


class _OpenAIModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class OpenAIFunction(_OpenAIModel):
    name: StrictStr = Field(min_length=1)
    description: StrictStr | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAITool(_OpenAIModel):
    type: Literal["function"] = "function"
    function: OpenAIFunction


class OpenAIToolChoiceFunction(_OpenAIModel):
    name: StrictStr = Field(min_length=1)


class OpenAIToolChoice(_OpenAIModel):
    type: Literal["function"] = "function"
    function: OpenAIToolChoiceFunction


class OpenAIToolCallFunction(_OpenAIModel):
    name: StrictStr = Field(min_length=1)
    arguments: StrictStr

    @field_validator("arguments")
    @classmethod
    def _arguments_are_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("tool call arguments must be valid JSON") from error
        return value


class OpenAIToolCall(_OpenAIModel):
    id: StrictStr
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


class OpenAIMediaURL(_OpenAIModel):
    url: StrictStr = Field(min_length=1)


class OpenAITextContentPart(_OpenAIModel):
    type: Literal["text"]
    text: StrictStr


class OpenAIImageURLContentPart(_OpenAIModel):
    type: Literal["image_url"]
    image_url: OpenAIMediaURL


class OpenAIInputImageContentPart(_OpenAIModel):
    type: Literal["input_image"]
    image_url: StrictStr = Field(min_length=1)


class OpenAIVideoURLContentPart(_OpenAIModel):
    type: Literal["video_url"]
    video_url: OpenAIMediaURL


class OpenAIInputVideoContentPart(_OpenAIModel):
    type: Literal["input_video"]
    video_url: StrictStr = Field(min_length=1)


OpenAIContentPart = (
    OpenAITextContentPart
    | OpenAIImageURLContentPart
    | OpenAIInputImageContentPart
    | OpenAIVideoURLContentPart
    | OpenAIInputVideoContentPart
)


class OpenAIMessage(_OpenAIModel):
    role: Literal[
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
        "latest_reminder",
    ]
    content: StrictStr | list[OpenAIContentPart] | None = Field(
        default="",
        discriminator=None,
    )
    reasoning_content: StrictStr | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: StrictStr | None = None

    @model_validator(mode="after")
    def _null_content_only_for_tool_calls(self) -> "OpenAIMessage":
        if self.content is None and (self.role != "assistant" or not self.tool_calls):
            raise ValueError(
                "content may be null only for assistant tool-call messages"
            )
        return self


class StreamOptions(_OpenAIModel):
    include_usage: StrictBool = False


class OpenAIThinking(_OpenAIModel):
    """Mainstream thinking switch (Kimi/GLM/Anthropic style).

    `budget_tokens` is accepted for Anthropic-style requests; the adapters
    do not implement thinking budgets, so the value is validated then unused.
    `keep` is the Kimi thinking-history passthrough flag; history handling is
    adapter-side, so the value is likewise validated then unused.
    """

    type: Literal["enabled", "disabled"]
    budget_tokens: StrictInt | None = Field(default=None, ge=1)
    keep: StrictStr | None = None


class ChatCompletionRequest(_OpenAIModel):
    model: StrictStr = Field(min_length=1)
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: StrictBool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: StrictInt | None = Field(default=None, ge=0)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=0)
    stop: StrictStr | list[StrictStr] | None = None
    thinking: OpenAIThinking | StrictBool | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "max"] | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: Literal["none", "auto", "required"] | OpenAIToolChoice | None = None
    parallel_tool_calls: StrictBool = True
    response_format: dict[str, Any] | None = None
    stream_options: StreamOptions | None = None
    n: StrictInt = 1
    logprobs: StrictBool = False
    top_logprobs: StrictInt | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    no_repeat_ngram_size: StrictInt = Field(default=0, ge=0)

    @field_validator("n")
    @classmethod
    def _one_choice_only(cls, value: int) -> int:
        if value != 1:
            raise ValueError("only n=1 is supported")
        return value

    @field_validator("logprobs")
    @classmethod
    def _logprobs_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("logprobs are not supported")
        return value

    @field_validator("top_logprobs")
    @classmethod
    def _top_logprobs_disabled(cls, value: int | None) -> None:
        if value is not None:
            raise ValueError("top_logprobs are not supported")
        return None

    @field_validator("frequency_penalty")
    @classmethod
    def _openai_penalties_must_be_zero(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("only a zero penalty is supported")
        return value

    @field_validator("messages")
    @classmethod
    def _validate_role_order(
        cls,
        messages: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        if messages[0].role in {"assistant", "tool", "latest_reminder"}:
            raise PydanticCustomError(
                "role_order",
                "the first message must be system, developer, or user",
            )
        if any(message.role == "system" for message in messages[1:]):
            raise PydanticCustomError(
                "role_order",
                "system messages are only allowed at the beginning",
            )

        pending_tool_ids: set[str] = set()
        for message in messages:
            if pending_tool_ids and message.role != "tool":
                raise PydanticCustomError(
                    "role_order",
                    "all assistant tool calls need tool results before the next message",
                )
            if message.role == "assistant":
                pending_tool_ids = {call.id for call in (message.tool_calls or ())}
            elif message.role == "tool":
                if not pending_tool_ids:
                    raise PydanticCustomError(
                        "role_order",
                        "tool messages must follow assistant tool calls",
                    )
                if (
                    message.tool_call_id is None
                    or message.tool_call_id not in pending_tool_ids
                ):
                    raise PydanticCustomError(
                        "role_order",
                        "tool_call_id must match a preceding assistant tool call",
                    )
                pending_tool_ids.remove(message.tool_call_id)

        if pending_tool_ids:
            raise PydanticCustomError(
                "role_order",
                "all assistant tool calls need tool results before generation",
            )
        if messages[-1].role in {"system", "assistant"}:
            raise PydanticCustomError(
                "role_order",
                "the final message must request an assistant response",
            )
        return messages

    @model_validator(mode="after")
    def _validate_combined_fields(self) -> "ChatCompletionRequest":
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise PydanticCustomError(
                "conflicting_max_tokens",
                "max_tokens and max_completion_tokens must match when both are supplied",
            )
        return self

    @property
    def effective_max_tokens(self) -> int | None:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens


@dataclass(frozen=True)
class OpenAIError(Exception):
    message: str
    status_code: int = 400
    error_type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None
    headers: dict[str, str] | None = None


def _error_response(error: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        headers=error.headers,
        content={
            "error": {
                "message": error.message,
                "type": error.error_type,
                "param": error.param,
                "code": error.code,
            }
        },
    )


def _validation_param(error: dict[str, Any]) -> str | None:
    error_type = error.get("type")
    if error_type == "conflicting_max_tokens":
        return "max_tokens"
    location = [
        str(part) for part in error.get("loc", ()) if part not in {"body", "__root__"}
    ]
    return ".".join(location) or None


def _validation_code(error: dict[str, Any]) -> str | None:
    if error.get("type") == "role_order":
        return "invalid_message_order"
    return None


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OpenAIError)
    async def handle_openai_error(
        _request: Request,
        error: OpenAIError,
    ) -> JSONResponse:
        return _error_response(error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        param = _validation_param(first)
        message = str(first.get("msg", "Invalid request"))
        if param is not None:
            message = f"Invalid value for '{param}': {message}"
        return _error_response(
            OpenAIError(
                message=message,
                param=param,
                code=_validation_code(first),
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        codes = {
            404: "not_found",
            405: "method_not_allowed",
        }
        return _error_response(
            OpenAIError(
                message=str(error.detail),
                status_code=error.status_code,
                code=codes.get(error.status_code, "http_error"),
                headers=error.headers,
            )
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        _request: Request,
        _error: Exception,
    ) -> JSONResponse:
        return _error_response(
            OpenAIError(
                message="The server encountered an internal error.",
                status_code=500,
                error_type="server_error",
                code="internal_server_error",
            )
        )


def _authentication_error(request: Request) -> OpenAIError | None:
    api_key = request.app.state.api_key
    if api_key is None:
        return None
    expected = f"Bearer {api_key}"
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied, expected):
        return OpenAIError(
            message="Incorrect API key provided.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


def install_authentication(app: FastAPI) -> None:
    @app.middleware("http")
    async def authenticate_v1_requests(
        request: Request,
        call_next: Any,
    ) -> Any:
        protected = request.url.path == "/v1" or request.url.path.startswith("/v1/")
        if protected:
            error = _authentication_error(request)
            if error is not None:
                return _error_response(error)
        return await call_next(request)


def _tool_call_from_openai(call: OpenAIToolCall) -> ToolCall:
    return ToolCall(
        id=call.id,
        type=call.type,
        function=ToolFunction(
            name=call.function.name,
            arguments=call.function.arguments,
        ),
    )


def _content_part_from_openai(part: OpenAIContentPart) -> ChatContentPart:
    if isinstance(part, OpenAITextContentPart):
        return ChatContentPart(type="text", text=part.text)
    if isinstance(part, OpenAIImageURLContentPart):
        return ChatContentPart(type=part.type, url=part.image_url.url)
    if isinstance(part, OpenAIInputImageContentPart):
        return ChatContentPart(type=part.type, url=part.image_url)
    if isinstance(part, OpenAIVideoURLContentPart):
        return ChatContentPart(type=part.type, url=part.video_url.url)
    if isinstance(part, OpenAIInputVideoContentPart):
        return ChatContentPart(type=part.type, url=part.video_url)
    raise TypeError(f"unsupported OpenAI content part: {type(part).__name__}")


def _message_from_openai(message: OpenAIMessage) -> ChatMessage:
    if isinstance(message.content, list):
        content_parts = tuple(
            _content_part_from_openai(part) for part in message.content
        )
        content = "".join(
            part.text or "" for part in content_parts if part.type == "text"
        )
    else:
        content_parts = ()
        content = "" if message.content is None else message.content
    return ChatMessage(
        role=message.role,
        content=content,
        content_parts=content_parts,
        reasoning_content=message.reasoning_content,
        tool_calls=tuple(
            _tool_call_from_openai(call) for call in (message.tool_calls or ())
        ),
        tool_call_id=message.tool_call_id,
    )


_SUPPORTED_EFFORTS = frozenset({"low", "medium", "high", "max"})


def _reasoning_options(
    service: ChatService,
    request: ChatCompletionRequest,
) -> tuple[str, str | None]:
    configured = getattr(service, "default_reasoning", None)

    effort = request.reasoning_effort
    if effort == "minimal":
        # OpenAI minimal 档映射到适配器支持的最低档。
        effort = "low"

    enabled: bool | None = None
    thinking = request.thinking
    if isinstance(thinking, OpenAIThinking):
        enabled = thinking.type == "enabled"
    elif thinking is not None:
        enabled = thinking

    if enabled is None:
        # 未显式传 thinking 开关：reasoning_effort 取 none 视为关闭，
        # 其余档位视为开启；都没有时回落到服务启动默认值。
        if effort == "none":
            enabled = False
        elif effort is not None:
            enabled = True
        elif configured is True or configured in _SUPPORTED_EFFORTS:
            enabled = True
            if configured in _SUPPORTED_EFFORTS:
                effort = configured
        else:
            enabled = False
    elif effort == "none":
        effort = None

    if not enabled:
        return "chat", None
    return "thinking", effort


def _options_from_openai(
    service: ChatService,
    request: ChatCompletionRequest,
) -> ChatOptions:
    thinking_mode, reasoning_effort = _reasoning_options(service, request)
    if request.stop is None:
        stop: tuple[str, ...] = ()
    elif isinstance(request.stop, str):
        stop = (request.stop,)
    else:
        stop = tuple(request.stop)

    tools = (
        ()
        if request.tools is None or request.tool_choice == "none"
        else tuple(tool.model_dump(exclude_none=True) for tool in request.tools)
    )
    tool_choice: object = request.tool_choice
    if isinstance(tool_choice, OpenAIToolChoice):
        tool_choice = tool_choice.model_dump(exclude_none=True)

    return ChatOptions(
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new=request.effective_max_tokens,
        stop=stop,
        repetition_penalty=request.repetition_penalty,
        presence_penalty=request.presence_penalty,
        no_repeat_ngram_size=request.no_repeat_ngram_size,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=request.response_format,
    )


def _tool_call_to_openai(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": call.type,
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }


def _completion_response(result: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.output.content,
    }
    if result.output.reasoning_content is not None:
        message["reasoning_content"] = result.output.reasoning_content
    if result.output.tool_calls:
        message["tool_calls"] = [
            _tool_call_to_openai(call) for call in result.output.tool_calls
        ]

    return {
        "id": result.request_id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": (result.prompt_tokens + result.completion_tokens),
        },
    }


def _completion_chunk(
    result: Any,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": result.request_id,
        "object": "chat.completion.chunk",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage(result: Any) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _completion_stream_payloads(
    result: Any,
    *,
    include_usage: bool,
) -> Iterator[dict[str, Any]]:
    yield _completion_chunk(
        result,
        {},
        finish_reason=result.finish_reason,
    )
    if include_usage:
        yield {
            "id": result.request_id,
            "object": "chat.completion.chunk",
            "created": result.created,
            "model": result.model,
            "choices": [],
            "usage": _usage(result),
        }


def _stream_delta_payload(
    ready: GenerationReady,
    delta: StreamDelta,
    *,
    tool_call_index: int,
) -> dict[str, Any]:
    if delta.kind == "reasoning":
        value = {"reasoning_content": delta.text}
    elif delta.kind == "content":
        value = {"content": delta.text}
    elif delta.kind == "tool_calls":
        value = {
            "tool_calls": [
                {
                    "index": tool_call_index + offset,
                    **_tool_call_to_openai(call),
                }
                for offset, call in enumerate(delta.tool_calls)
            ]
        }
    else:
        raise ValueError(f"unsupported stream delta kind: {delta.kind!r}")
    return _completion_chunk(ready, value)


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ready_completion_chunk(ready: GenerationReady) -> dict[str, Any]:
    return {
        "id": ready.request_id,
        "object": "chat.completion.chunk",
        "created": ready.created,
        "model": ready.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }


@dataclass
class _CompletionStreamState:
    request: Request
    events: asyncio.Queue[tuple[str, object]]
    cancellation: threading.Event
    worker: asyncio.Task[None]


def _start_completion_stream(
    request: Request,
    service: ChatService,
    messages: list[ChatMessage],
    options: ChatOptions,
    *,
    include_usage: bool,
) -> _CompletionStreamState:
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    cancellation = threading.Event()

    def emit(event: tuple[str, object]) -> None:
        loop.call_soon_threadsafe(events.put_nowait, event)

    def complete_in_worker() -> None:
        try:
            ready_value: GenerationReady | None = None
            next_tool_call_index = 0

            def on_ready(ready: GenerationReady) -> None:
                nonlocal ready_value
                ready_value = ready
                emit(("ready", ready))

            def on_stream_delta(delta: StreamDelta) -> None:
                nonlocal next_tool_call_index
                if ready_value is None:
                    raise RuntimeError(
                        "stream delta arrived before generation readiness"
                    )
                emit(
                    (
                        "payload",
                        _stream_delta_payload(
                            ready_value,
                            delta,
                            tool_call_index=next_tool_call_index,
                        ),
                    )
                )
                next_tool_call_index += len(delta.tool_calls)

            result = service.complete(
                messages,
                options,
                cancel_event=cancellation,
                on_ready=on_ready,
                on_stream_delta=on_stream_delta,
            )
            for payload in _completion_stream_payloads(
                result,
                include_usage=include_usage,
            ):
                emit(("payload", payload))
        except BaseException as error:
            emit(("error", error))
        else:
            emit(("done", None))

    worker = asyncio.create_task(asyncio.to_thread(complete_in_worker))
    return _CompletionStreamState(
        request=request,
        events=events,
        cancellation=cancellation,
        worker=worker,
    )


async def _finish_completion_stream(
    state: _CompletionStreamState,
) -> None:
    if not state.worker.done():
        state.cancellation.set()
    try:
        await asyncio.shield(state.worker)
    except asyncio.CancelledError:
        state.cancellation.set()
        try:
            await state.worker
        except BaseException:
            pass
        raise


async def _next_completion_stream_event(
    state: _CompletionStreamState,
    *,
    idle_timeout: float | None = None,
) -> tuple[str, object]:
    waited = 0.0
    while True:
        if await state.request.is_disconnected():
            state.cancellation.set()
            await asyncio.shield(state.worker)
            return ("disconnect", None)
        try:
            return await asyncio.wait_for(
                state.events.get(),
                timeout=0.05,
            )
        except asyncio.TimeoutError:
            waited += 0.05
            if idle_timeout is not None and waited >= idle_timeout:
                return ("idle", None)
            continue


async def _prepare_completion_stream(
    request: Request,
    service: ChatService,
    messages: list[ChatMessage],
    options: ChatOptions,
    *,
    include_usage: bool,
) -> tuple[_CompletionStreamState, tuple[str, object]]:
    state = _start_completion_stream(
        request,
        service,
        messages,
        options,
        include_usage=include_usage,
    )
    try:
        first_event = await _next_completion_stream_event(state)
        kind, value = first_event
        if kind == "error":
            if not isinstance(value, BaseException):
                raise RuntimeError("invalid streaming error event")
            await _finish_completion_stream(state)
            raise value
        if kind not in {"ready", "disconnect"}:
            await _finish_completion_stream(state)
            raise RuntimeError("invalid initial streaming event")
        return state, first_event
    except BaseException:
        if not state.worker.done():
            state.cancellation.set()
            try:
                await asyncio.shield(state.worker)
            except BaseException:
                pass
        raise


_STREAM_HEARTBEAT_S = 15.0


async def _chat_completion_stream(
    state: _CompletionStreamState,
    first_event: tuple[str, object],
    *,
    heartbeat_s: float = _STREAM_HEARTBEAT_S,
) -> AsyncIterator[str]:
    event = first_event
    try:
        while True:
            kind, value = event
            if kind == "disconnect":
                return
            if kind == "done":
                break
            if kind == "error":
                # HTTP headers are already committed. Closing without [DONE]
                # is the only safe way to signal an incomplete SSE response.
                if isinstance(value, BaseException):
                    print(
                        "[cccp-stream] generation failed after SSE headers were sent:",
                        flush=True,
                    )
                    traceback.print_exception(value)
                return
            if kind == "idle":
                # SSE 注释心跳：长 prefill 期间保持连接活跃，
                # 防止客户端读取超时断开。
                yield ": ping\n\n"
                event = await _next_completion_stream_event(
                    state,
                    idle_timeout=heartbeat_s,
                )
                continue
            if kind == "ready":
                if not isinstance(value, GenerationReady):
                    return
                yield _sse_frame(_ready_completion_chunk(value))
                event = await _next_completion_stream_event(
                    state,
                    idle_timeout=heartbeat_s,
                )
                continue
            if kind != "payload" or not isinstance(value, dict):
                return
            yield _sse_frame(value)
            event = await _next_completion_stream_event(
                state,
                idle_timeout=heartbeat_s,
            )
        yield "data: [DONE]\n\n"
    finally:
        await _finish_completion_stream(state)


async def _wait_for_http_disconnect(request: Request) -> None:
    """Wait on the ASGI receive channel after FastAPI consumed the body."""
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _complete_nonstream(
    request: Request,
    service: ChatService,
    messages: list[ChatMessage],
    options: ChatOptions,
) -> Any:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            service.complete,
            messages,
            options,
            cancel_event=cancellation,
        )
    )
    disconnect = asyncio.create_task(_wait_for_http_disconnect(request))
    try:
        done, _pending = await asyncio.wait(
            {worker, disconnect},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker in done:
            return await worker
        await disconnect
        cancellation.set()
        await asyncio.shield(worker)
        raise HTTPException(
            status_code=499,
            detail="client disconnected",
        )
    except asyncio.CancelledError:
        cancellation.set()
        try:
            await asyncio.shield(worker)
        except BaseException:
            pass
        raise
    except BaseException:
        if not worker.done():
            cancellation.set()
            try:
                await asyncio.shield(worker)
            except BaseException:
                pass
        raise
    finally:
        if not disconnect.done():
            disconnect.cancel()
        try:
            await disconnect
        except BaseException:
            pass


_MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "kimi_k3": {
        "tool_use": True,
        "efforts": ("low", "medium", "high", "max"),
        "default_effort": "max",
        "image": True,
        "video": True,
    },
    "dsv4": {
        "tool_use": True,
        "efforts": ("high", "max"),
        "default_effort": "high",
        "image": False,
        "video": False,
    },
    "glm": {
        "tool_use": False,
        "efforts": (),
        "default_effort": None,
        "image": False,
        "video": False,
    },
    "glm5_next": {
        "tool_use": False,
        "efforts": (),
        "default_effort": None,
        "image": True,
        "video": False,
    },
    "qwen3_5_dense": {
        "tool_use": True,
        "efforts": ("low", "medium", "high", "max"),
        "default_effort": "max",
        "image": False,
        "video": False,
    },
}


def _served_architecture(service: object) -> str:
    return (
        getattr(service, "architecture", None)
        or getattr(getattr(service, "engine", None), "arch", None)
        or getattr(getattr(service, "adapter", None), "name", None)
        or ""
    )


def _input_capability(
    service: object,
    capabilities: dict[str, Any],
    medium: Literal["image", "video"],
) -> bool:
    names = (f"supports_{medium}_in", f"supports_{medium}")
    sources = (
        service,
        getattr(service, "model", None),
        getattr(service, "engine", None),
    )
    for source in sources:
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if isinstance(value, bool):
                return value
        advertised = getattr(source, "capabilities", None)
        if isinstance(advertised, dict):
            for name in (*names, medium):
                value = advertised.get(name)
                if isinstance(value, bool):
                    return value
    return bool(capabilities.get(medium, False))


def _model_payload(request: Request) -> dict[str, Any]:
    service = request.app.state.chat_service
    capabilities = _MODEL_CAPABILITIES.get(_served_architecture(service), {})
    tool_use = bool(capabilities.get("tool_use"))
    image_input = _input_capability(service, capabilities, "image")
    video_input = _input_capability(service, capabilities, "video")
    efforts = capabilities.get("efforts", ())
    context_length = request.app.state.model_context_length
    input_modalities = ["text"]
    if image_input:
        input_modalities.append("image")
    if video_input:
        input_modalities.append("video")
    payload: dict[str, Any] = {
        "id": request.app.state.served_model_name,
        "object": "model",
        "created": request.app.state.model_created,
        "owned_by": "cccp",
        # Kimi 平台模型目录字段：kimi-code 据此渲染思考开关与档位选择。
        "display_name": request.app.state.served_model_name,
        "supports_reasoning": True,
        "supports_image_in": image_input,
        "supports_video_in": video_input,
        "supports_tool_use": tool_use,
        "supports_thinking_type": "both",
        # 自定义注册表（api.json）风格的扁平字段。
        "tool_call": tool_use,
        "reasoning": True,
        "modalities": {"input": input_modalities, "output": ["text"]},
    }
    if context_length is not None:
        payload["context_length"] = context_length
        payload["limit"] = {"context": context_length}
    if efforts:
        payload["think_efforts"] = {
            "support": True,
            "valid_efforts": list(efforts),
            "default_effort": capabilities["default_effort"],
        }
        payload["support_efforts"] = list(efforts)
        payload["default_effort"] = capabilities["default_effort"]
    return payload


def _model_not_found(model: str) -> OpenAIError:
    return OpenAIError(
        message=(
            f"The model `{model}` does not exist "
            "or you do not have access to it."
        ),
        status_code=404,
        param="model",
        code="model_not_found",
    )


def _runtime_objects(service: object) -> tuple[object | None, object | None, object | None]:
    engine = getattr(service, "engine", None)
    model = getattr(engine, "model", None)
    return model, getattr(model, "store", None), getattr(model, "pool", None)


def _declared_expert_count(store: object | None) -> int:
    if store is None:
        return 0
    manifest = getattr(store, "man", None)
    layers = getattr(manifest, "expert_files", {}) or {}
    config = getattr(store, "cfg", {}) or {}
    default_count = int(config.get("n_experts") or 0)
    tiers = getattr(manifest, "tiers_per_layer", {}) or {}
    total = 0
    for layer in layers:
        layer_tiers = tiers.get(int(layer), tiers.get(str(layer)))
        if isinstance(layer_tiers, str):
            total += sum(value.lower() != "d" for value in layer_tiers)
        else:
            total += default_count
    return total


def _mapping_keys(value: object) -> set[object]:
    if not isinstance(value, dict):
        return set()
    return set(value)


def _runtime_receipt(service: object) -> dict[str, Any]:
    """Return the I-5 profile acceptance and expert residency receipt."""
    model, store, pool = _runtime_objects(service)
    root = getattr(store, "root", None)
    environment_profile = os.environ.get("CCCP_PROFILE_JSON")
    model_profile = None
    if root:
        candidate = os.path.join(str(root), "profile.json")
        if os.path.isfile(candidate):
            model_profile = candidate
    requested_profile = environment_profile or model_profile
    loaded_profile = getattr(store, "profile_path", None)

    pinned = _mapping_keys(getattr(pool, "pinned", None))
    cache = _mapping_keys(getattr(pool, "cache", None))
    device = getattr(model, "device", None)
    device_type = getattr(device, "type", str(device or "cpu"))
    full_resident = bool(getattr(pool, "full_resident", False))
    declared = _declared_expert_count(store)
    if full_resident and device_type != "cpu":
        ram_keys = pinned
        vram_count = declared
    elif device_type == "cpu":
        ram_keys = pinned | cache
        vram_count = 0
    else:
        ram_keys = pinned
        vram_count = len(cache)

    return {
        "plan_file_requested": bool(requested_profile),
        "plan_file_ok": (
            bool(loaded_profile) if requested_profile else True
        ),
        "plan_file_name": (
            os.path.basename(str(loaded_profile or requested_profile))
            if requested_profile
            else None
        ),
        "experts_declared": declared,
        "experts_resident_ram": len(ram_keys),
        "experts_resident_vram": vram_count,
        "expert_resident_ram_bytes": int(
            getattr(pool, "host_expert_bytes", 0) or 0
        ),
        "expert_resident_vram_bytes": int(
            getattr(pool, "gpu_storage_bytes", 0) or 0
        ),
    }


def _router_stats_payload(service: object) -> dict[str, Any]:
    _model, store, pool = _runtime_objects(service)
    raw_counts = getattr(pool, "route_counts", None)
    counts = {
        f"{int(layer)}:{int(expert)}": int(value)
        for (layer, expert), value in (raw_counts or {}).items()
    }
    config = getattr(store, "cfg", {}) or {}
    layers = getattr(getattr(store, "man", None), "expert_files", {}) or {}
    observations_per_token = len(layers) * int(config.get("top_k") or 0)
    started_at = getattr(pool, "route_counts_started_at", None)
    return {
        "schema": "cccp-router-stats-v1",
        "enabled": os.environ.get("CCCP_ROUTE_COUNTS", "0") != "0",
        "counts": counts,
        "tokens_seen": (
            sum(counts.values()) // observations_per_token
            if observations_per_token
            else 0
        ),
        "wall_time_s": (
            max(0.0, time.monotonic() - float(started_at))
            if started_at is not None
            else 0.0
        ),
    }


def install_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        service = request.app.state.chat_service
        architecture = _served_architecture(service) or "unknown"
        return {
            "status": "ok",
            "ready": True,
            "model": request.app.state.served_model_name,
            "architecture": architecture,
            "busy": bool(getattr(service, "busy", False)),
            **_runtime_receipt(service),
        }

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [_model_payload(request)],
        }

    @app.get("/v1/models/{model_id}")
    def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
        if model_id != request.app.state.served_model_name:
            raise _model_not_found(model_id)
        return _model_payload(request)

    @app.get("/v1/model-spec")
    def model_spec(request: Request) -> dict[str, Any]:
        service = request.app.state.chat_service
        _model, store, _pool = _runtime_objects(service)
        root = getattr(store, "root", None)
        if not root:
            raise HTTPException(status_code=503, detail="model metadata unavailable")
        from .interface_contract import model_spec_payload

        return model_spec_payload(root)

    @app.get("/v1/expert-bytes")
    def expert_bytes(request: Request) -> dict[str, Any]:
        service = request.app.state.chat_service
        _model, store, _pool = _runtime_objects(service)
        root = getattr(store, "root", None)
        if not root:
            raise HTTPException(status_code=503, detail="model metadata unavailable")
        from .interface_contract import expert_bytes_payload

        return expert_bytes_payload(root)

    @app.get("/v1/expert-stats")
    def expert_stats(request: Request) -> dict[str, Any]:
        return _router_stats_payload(request.app.state.chat_service)

    @app.post("/v1/expert-stats/reset")
    def reset_expert_stats(request: Request) -> dict[str, Any]:
        service = request.app.state.chat_service
        if bool(getattr(service, "busy", False)):
            raise HTTPException(status_code=409, detail="model is busy")
        _model, _store, pool = _runtime_objects(service)
        if pool is None:
            raise HTTPException(status_code=503, detail="expert pool unavailable")
        os.environ["CCCP_ROUTE_COUNTS"] = "1"
        pool.route_counts = Counter()
        pool.route_counts_started_at = time.monotonic()
        return _router_stats_payload(service)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request_body: ChatCompletionRequest,
        request: Request,
    ) -> Any:
        served_model_name = request.app.state.served_model_name
        if request_body.model != served_model_name:
            raise _model_not_found(request_body.model)

        service = request.app.state.chat_service
        messages = [_message_from_openai(message) for message in request_body.messages]
        options = _options_from_openai(service, request_body)
        try:
            if request_body.stream:
                include_usage = bool(
                    request_body.stream_options
                    and request_body.stream_options.include_usage
                )
                state, first_event = await _prepare_completion_stream(
                    request,
                    service,
                    messages,
                    options,
                    include_usage=include_usage,
                )
                return StreamingResponse(
                    _chat_completion_stream(state, first_event),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            result = await _complete_nonstream(
                request,
                service,
                messages,
                options,
            )
        except ChatQueueFull as error:
            raise OpenAIError(
                message=str(error),
                status_code=429,
                error_type="rate_limit_error",
                code="queue_full",
            ) from error
        except UnsupportedChatCapability as error:
            raise OpenAIError(
                message=str(error),
                param="messages",
                code="unsupported_chat_capability",
            ) from error
        return _completion_response(result)


def create_app(
    service: ChatService,
    *,
    served_model_name: str,
    api_key: str | None,
    cors_allow_origins: tuple[str, ...] = (),
    context_length: int | None = None,
) -> FastAPI:
    app = FastAPI(
        title="CCCP OpenAI API",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    if cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.state.chat_service = service
    app.state.served_model_name = served_model_name
    app.state.model_created = int(time.time())
    app.state.model_context_length = context_length
    app.state.api_key = api_key
    install_error_handlers(app)
    install_routes(app)
    install_authentication(app)
    return app


__all__ = [
    "ChatCompletionRequest",
    "OpenAIError",
    "OpenAIMessage",
    "OpenAIThinking",
    "OpenAITool",
    "OpenAIToolChoice",
    "StreamOptions",
    "create_app",
    "install_authentication",
    "install_error_handlers",
    "install_routes",
]
