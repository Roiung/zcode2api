"""OpenAI Chat Completions 兼容层。

将 OpenAI `/v1/chat/completions` 请求转换为内部 Anthropic Messages 请求，
复用现有 `/v1/messages` 主链路（账号轮询、验证码、换号、SSE 透传）。
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..auth_admin import _extract_bearer, verify_gateway_key
from ..store import store
from .gateway import messages as anthropic_messages

router = APIRouter()


def _json_string(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _pick_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join([p for p in parts if p])
    return ""


def _to_anthropic_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    blocks.append({"type": "text", "text": text})
        if blocks:
            return blocks
    return [{"type": "text", "text": _pick_text(content)}]


def _build_system(payload: dict):
    system = payload.get("system")
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        blocks = []
        for item in system:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text") and isinstance(item.get("text"), str):
                blocks.append({"type": "text", "text": item["text"]})
        return blocks or None
    return None


def _map_tools(payload: dict):
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return None

    mapped = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function":
            fn = item.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            mapped.append(
                {
                    "name": name,
                    "description": fn.get("description") or item.get("description") or "",
                    "input_schema": fn.get("parameters") or item.get("parameters") or {"type": "object", "properties": {}},
                }
            )
            continue
        if item_type == "tool":
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            mapped.append(
                {
                    "name": name,
                    "description": item.get("description") or "",
                    "input_schema": item.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    return mapped or None


def _map_tool_choice(payload: dict):
    tool_choice = payload.get("tool_choice")
    if tool_choice is None:
        tool_choice = payload.get("text", {}).get("tool_choice") if isinstance(payload.get("text"), dict) else None
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "tool", "name": name}
        if tool_choice.get("type") == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "tool", "name": name}
    return None


def _map_openai_message(msg: dict) -> dict | None:
    role = msg.get("role") or "user"
    if role == "system":
        return None

    content = _to_anthropic_content(msg.get("content"))
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        blocks = []
        for block in content:
            if block.get("text"):
                blocks.append(block)
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "function":
                fn = item.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments")
            elif item_type in ("tool_call", "tool"):
                name = item.get("name")
                raw_args = item.get("arguments") or item.get("input")
            else:
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("id") or f"toolu_{uuid.uuid4().hex}",
                    "name": name,
                    "input": parsed_args,
                }
            )
        content = blocks or [{"type": "text", "text": ""}]

    if role in ("tool", "function"):
        tool_call_id = msg.get("tool_call_id") or msg.get("call_id")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id or f"toolu_{uuid.uuid4().hex}",
                    "content": _pick_text(msg.get("content") or msg.get("output")),
                }
            ],
        }

    return {"role": role, "content": content}


def openai_to_anthropic(payload: dict) -> dict:
    body: dict = {
        "model": payload.get("model"),
        "messages": [],
        "stream": bool(payload.get("stream", False)),
    }

    if payload.get("max_tokens") is not None:
        body["max_tokens"] = payload.get("max_tokens")
    elif payload.get("max_completion_tokens") is not None:
        body["max_tokens"] = payload.get("max_completion_tokens")

    for key in ("temperature", "top_p", "top_k", "metadata"):
        value = payload.get(key)
        if value is not None:
            body[key] = value

    stop = payload.get("stop")
    if isinstance(stop, str):
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        body["stop_sequences"] = [s for s in stop if isinstance(s, str)]

    system = _build_system(payload)
    if system:
        body["system"] = system

    tools = _map_tools(payload)
    if tools:
        body["tools"] = tools

    tool_choice = _map_tool_choice(payload)
    if tool_choice:
        body["tool_choice"] = tool_choice

    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        mapped = _map_openai_message(msg)
        if mapped:
            body["messages"].append(mapped)

    return body


def _response_input_to_messages(payload: dict) -> list[dict]:
    raw_input = payload.get("input")
    if raw_input is None:
        return payload.get("messages") or []
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if isinstance(raw_input, list):
        messages: list[dict] = []
        current_assistant: dict | None = None
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                role = item.get("role") or "user"
                content = item.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                            text = part.get("text")
                            if isinstance(text, str):
                                parts.append({"type": "text", "text": text})
                    content = parts if parts else content
                messages.append({"role": role, "content": content})
                current_assistant = messages[-1] if role == "assistant" else None
            elif item_type in ("function_call", "tool_call"):
                if current_assistant is None:
                    current_assistant = {"role": "assistant", "content": None, "tool_calls": []}
                    messages.append(current_assistant)
                current_assistant.setdefault("tool_calls", []).append(
                    {
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "tool",
                            "arguments": item.get("arguments") if isinstance(item.get("arguments"), str) else _json_string(item.get("arguments") or item.get("input") or {}),
                        },
                    }
                )
            elif item_type in ("function_call_output", "tool_result"):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id") or item.get("tool_call_id") or f"call_{uuid.uuid4().hex}",
                        "content": item.get("output") if isinstance(item.get("output"), str) else _json_string(item.get("output") or item.get("content") or ""),
                    }
                )
        return messages
    return []


def _responses_payload_to_chat_payload(payload: dict) -> dict:
    chat_payload = dict(payload)
    chat_payload["messages"] = _response_input_to_messages(payload)
    if payload.get("max_output_tokens") is not None and payload.get("max_tokens") is None:
        chat_payload["max_tokens"] = payload.get("max_output_tokens")
    if payload.get("tool_choice") is None and isinstance(payload.get("text"), dict):
        if payload["text"].get("tool_choice") is not None:
            chat_payload["tool_choice"] = payload["text"].get("tool_choice")
    return chat_payload


def _response_output_from_message(message: dict) -> list[dict]:
    output: list[dict] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
        elif item_type == "tool_use":
            output.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or f"fc_{uuid.uuid4().hex}",
                    "call_id": item.get("id") or f"call_{uuid.uuid4().hex}",
                    "name": item.get("name") or "tool",
                    "arguments": _json_string(item.get("input") or {}),
                }
            )
    return output


def _collect_text_and_tools(message: dict) -> tuple[str | None, list[dict]]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif item_type == "tool_use":
            tool_calls.append(
                {
                    "id": item.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "tool",
                        "arguments": _json_string(item.get("input") or {}),
                    },
                }
            )
    text = "".join(text_parts)
    return (text if text else None), tool_calls


def anthropic_to_openai_json(data: dict, model: str) -> dict:
    message = data.get("message") or {}
    content_text, tool_calls = _collect_text_and_tools(message)

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("input_tokens") or 0
    completion_tokens = usage.get("output_tokens") or 0
    total_tokens = prompt_tokens + completion_tokens

    return {
        "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": "tool_calls" if tool_calls else (data.get("stop_reason") or "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _sse_bytes(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _gateway_token_from_request(request: Request) -> str | None:
    bearer = _extract_bearer(request.headers.get("authorization"))
    return bearer or request.headers.get("x-api-key")


def _check_gateway_key_or_error(request: Request) -> JSONResponse | None:
    key = store.gateway_key()
    if not key:
        return None
    token = _gateway_token_from_request(request)
    if token is None:
        return JSONResponse({"detail": "缺少 API Key"}, status_code=401)
    if token != key:
        return JSONResponse({"detail": "API Key 无效"}, status_code=403)
    return None


async def anthropic_stream_to_openai(resp, model: str):
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    role_sent = False
    done_sent = False
    tool_index = 0
    current_tool_meta: dict[int, str] = {}

    async for raw in resp.body_iterator:
        if not raw:
            continue
        text = raw.decode("utf-8", "ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                if not done_sent:
                    done_sent = True
                    yield b"data: [DONE]\n\n"
                continue
            try:
                event = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue

            event_type = event.get("type")
            if event_type == "message_start":
                if not role_sent:
                    yield _sse_bytes(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                        }
                    )
                    role_sent = True
            elif event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    idx = event.get("index")
                    if not isinstance(idx, int):
                        idx = tool_index
                        tool_index += 1
                    else:
                        tool_index = max(tool_index, idx + 1)
                    tool_id = block.get("id") or f"call_{uuid.uuid4().hex}"
                    current_tool_meta[idx] = tool_id
                    yield _sse_bytes(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": idx,
                                                "id": tool_id,
                                                "type": "function",
                                                "function": {
                                                    "name": block.get("name") or "tool",
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif event_type == "content_block_delta":
                delta = ((event.get("delta") or {}).get("text")) or ""
                if delta:
                    yield _sse_bytes(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                    )
            elif event_type == "input_json_delta":
                delta = ((event.get("delta") or {}).get("partial_json")) or ""
                idx = event.get("index")
                if delta and isinstance(idx, int):
                    yield _sse_bytes(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": idx,
                                                "id": current_tool_meta.get(idx) or f"call_{uuid.uuid4().hex}",
                                                "type": "function",
                                                "function": {
                                                    "arguments": delta,
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif event_type == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason") or "stop"
                yield _sse_bytes(
                    {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if stop_reason == "tool_use" else stop_reason}],
                    }
                )
            elif event_type == "message_stop":
                if not done_sent:
                    done_sent = True
                    yield b"data: [DONE]\n\n"

    if not done_sent:
        yield b"data: [DONE]\n\n"


def _responses_json_from_anthropic(data: dict, model: str) -> dict:
    message = data.get("message") or {}
    output = _response_output_from_message(message)
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("input_tokens") or 0
    completion_tokens = usage.get("output_tokens") or 0
    total_tokens = prompt_tokens + completion_tokens
    text_parts = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

    return {
        "id": data.get("id") or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "output_text": "".join(text_parts),
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


async def anthropic_stream_to_responses(resp, model: str):
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    output_index = 0
    current_tool_meta: dict[int, str] = {}

    yield _sse_bytes({"type": "response.created", "response": {"id": response_id, "object": "response", "created_at": created, "model": model, "status": "in_progress"}})

    async for raw in resp.body_iterator:
        if not raw:
            continue
        text = raw.decode("utf-8", "ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue

            event_type = event.get("type")
            if event_type == "message_start":
                yield _sse_bytes({"type": "response.in_progress", "response": {"id": response_id, "object": "response", "created_at": created, "model": model, "status": "in_progress"}})
            elif event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    idx = event.get("index")
                    if not isinstance(idx, int):
                        idx = output_index
                        output_index += 1
                    else:
                        output_index = max(output_index, idx + 1)
                    call_id = block.get("id") or f"call_{uuid.uuid4().hex}"
                    current_tool_meta[idx] = call_id
                    yield _sse_bytes({
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": block.get("name") or "tool",
                            "arguments": "",
                        },
                    })
            elif event_type == "content_block_delta":
                delta = ((event.get("delta") or {}).get("text")) or ""
                if delta:
                    yield _sse_bytes({"type": "response.output_text.delta", "delta": delta})
            elif event_type == "input_json_delta":
                delta = ((event.get("delta") or {}).get("partial_json")) or ""
                idx = event.get("index")
                if delta and isinstance(idx, int):
                    yield _sse_bytes({
                        "type": "response.function_call_arguments.delta",
                        "output_index": idx,
                        "item_id": current_tool_meta.get(idx) or f"call_{uuid.uuid4().hex}",
                        "delta": delta,
                    })
            elif event_type == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason") or "stop"
                if stop_reason == "tool_use":
                    yield _sse_bytes({"type": "response.output_item.done"})
            elif event_type == "message_stop":
                yield _sse_bytes({"type": "response.completed", "response": {"id": response_id, "object": "response", "created_at": created, "model": model, "status": "completed"}})
                yield b"data: [DONE]\n\n"
                return

    yield _sse_bytes({"type": "response.completed", "response": {"id": response_id, "object": "response", "created_at": created, "model": model, "status": "completed"}})
    yield b"data: [DONE]\n\n"


@router.post("/v1/responses", dependencies=[Depends(verify_gateway_key)])
async def responses_api(request: Request):
    auth_error = _check_gateway_key_or_error(request)
    if auth_error is not None:
        return auth_error
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": {"message": "请求体不是合法 JSON", "type": "invalid_request_error"}}, status_code=400)

    model = str(payload.get("model") or "")
    chat_payload = _responses_payload_to_chat_payload(payload)
    anthropic_body = openai_to_anthropic(chat_payload)
    request._body = json.dumps(anthropic_body).encode("utf-8")

    response = await anthropic_messages(request)
    if not isinstance(response, StreamingResponse):
        return response

    if not payload.get("stream", False):
        raw = bytearray()
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                raw.extend(chunk)
            elif isinstance(chunk, str):
                raw.extend(chunk.encode("utf-8"))
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return JSONResponse({"error": {"message": "上游返回非 JSON", "type": "upstream_error"}}, status_code=502)
        return JSONResponse(_responses_json_from_anthropic(data, model), status_code=response.status_code)

    headers = {"Cache-Control": "no-cache"}
    return StreamingResponse(
        anthropic_stream_to_responses(response, model),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/v1/chat/completions", dependencies=[Depends(verify_gateway_key)])
async def chat_completions(request: Request):
    auth_error = _check_gateway_key_or_error(request)
    if auth_error is not None:
        return auth_error
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": {"message": "请求体不是合法 JSON", "type": "invalid_request_error"}}, status_code=400)

    model = str(payload.get("model") or "")
    anthropic_body = openai_to_anthropic(payload)
    request._body = json.dumps(anthropic_body).encode("utf-8")

    response = await anthropic_messages(request)
    if not isinstance(response, StreamingResponse):
        return response

    if not payload.get("stream", False):
        raw = bytearray()
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                raw.extend(chunk)
            elif isinstance(chunk, str):
                raw.extend(chunk.encode("utf-8"))
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return JSONResponse({"error": {"message": "上游返回非 JSON", "type": "upstream_error"}}, status_code=502)
        return JSONResponse(anthropic_to_openai_json(data, model), status_code=response.status_code)

    headers = {"Cache-Control": "no-cache"}
    return StreamingResponse(
        anthropic_stream_to_openai(response, model),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers=headers,
    )