#!/usr/bin/env python3
"""
Tool-call conversion proxy for llama.cpp + Hermes Agent.
Listens on port 8088, forwards to llama.cpp on port 8089.
- Native OpenAI tool calls (e.g. Cline): streamed directly, no buffering.
- Text-format tool calls (e.g. Hermes): buffered, converted to OpenAI format.
"""
import json, re, uuid, asyncio
import aiohttp
from aiohttp import web

BACKEND   = "http://localhost:8089"
PORT      = 8088
HOST      = "0.0.0.0"

_TOOL_RES = [
    re.compile(r"<tools>\s*(.*?)\s*</tools>",         re.DOTALL),
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```",     re.DOTALL),
]

def _try_parse_call(text):
    try:
        d = json.loads(text.strip())
        if isinstance(d, dict) and "name" in d:
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": d["name"],
                    "arguments": json.dumps(d.get("arguments", {})),
                },
            }
    except Exception:
        pass
    return None

def _extract_json_objects(text):
    results = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                results.append(text[start:i+1])
                start = None
    return results

def _extract_tool_calls(content):
    for pat in _TOOL_RES:
        calls = [c for m in pat.findall(content) for c in [_try_parse_call(m)] if c]
        if calls:
            return calls
    for obj_str in _extract_json_objects(content):
        call = _try_parse_call(obj_str)
        if call:
            return [call]
    return []

def _transform_nonstream(data):
    for choice in data.get("choices", []):
        content = (choice.get("message") or {}).get("content") or ""
        calls = _extract_tool_calls(content)
        if calls:
            choice["message"]["tool_calls"] = calls
            choice["message"]["content"]    = None
            choice["finish_reason"]         = "tool_calls"
    return data

def _make_sse_tool_chunk(base_id, model, calls):
    chunk = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": i,
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": c["function"]["arguments"],
                        },
                    }
                    for i, c in enumerate(calls)
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

async def _stream_passthrough(request, resp):
    """Stream backend response directly to client with no buffering."""
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
    }
    response = web.StreamResponse(status=resp.status, headers=resp_headers)
    await response.prepare(request)
    try:
        async for chunk in resp.content.iter_any():
            await response.write(chunk)
    except Exception as e:
        print(f"[proxy] stream passthrough error: {e}", flush=True)
    await response.write_eof()
    return response

async def handle(request):
    path    = request.raw_path
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    body    = await request.read()

    client_ip = request.remote
    is_chat   = "/chat/completions" in request.path
    if is_chat:
        print(f"[proxy] {request.method} {request.path} from {client_ip}", flush=True)

    # Detect native OpenAI tool calling in request body
    has_native_tools = False
    is_stream = False
    if is_chat and body:
        try:
            req_json = json.loads(body)
            is_stream = bool(req_json.get("stream"))
            has_native_tools = bool(req_json.get("tools"))
        except Exception:
            pass

    if has_native_tools:
        print(f"[proxy] native tools detected — streaming passthrough", flush=True)

    try:
        async with aiohttp.ClientSession() as s:
            async with s.request(
                method=request.method,
                url=f"{BACKEND}{path}",
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as resp:

                # Native tool calls or non-chat: stream directly, no buffering
                if has_native_tools or not is_chat:
                    return await _stream_passthrough(request, resp)

                if is_chat and is_stream and resp.status == 200:
                    # Buffer the SSE stream for text-format tool call conversion
                    raw_lines   = []
                    content_acc = []
                    base_id     = "chatcmpl-proxy"
                    model_name  = "llama-server"
                    truncated   = False

                    try:
                        async for raw_line in resp.content:
                            line = raw_line.decode(errors="replace").rstrip()
                            raw_lines.append(line)
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    base_id    = chunk.get("id", base_id)
                                    model_name = chunk.get("model", model_name)
                                    for ch in chunk.get("choices", []):
                                        delta_content = (ch.get("delta") or {}).get("content")
                                        if delta_content:
                                            content_acc.append(delta_content)
                                except Exception:
                                    pass
                    except Exception as stream_err:
                        truncated = True
                        print(f"[proxy] stream truncated: {stream_err}", flush=True)

                    if truncated and not content_acc:
                        return web.Response(
                            status=503,
                            text="llama-server disconnected mid-stream (likely OOM). Try a shorter context or smaller ubatch.",
                        )

                    full_content = "".join(content_acc)
                    calls = _extract_tool_calls(full_content)

                    if calls:
                        print(f"[proxy] converted tool call: {[c['function']['name'] for c in calls]}", flush=True)
                        sse_body = _make_sse_tool_chunk(base_id, model_name, calls).encode()
                        return web.Response(
                            status=200, body=sse_body,
                            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                        )
                    else:
                        sse_body = ("\n".join(raw_lines) + "\n").encode()
                        return web.Response(
                            status=200, body=sse_body,
                            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                        )

                else:
                    raw = await resp.read()
                    resp_headers = {k: v for k, v in resp.headers.items()
                                    if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")}
                    if is_chat and resp.status == 200:
                        try:
                            raw = json.dumps(_transform_nonstream(json.loads(raw))).encode()
                        except Exception:
                            pass
                    return web.Response(status=resp.status, body=raw, headers=resp_headers)

    except aiohttp.ClientConnectorError:
        return web.Response(status=502, text="llama-server not reachable on port 8089")
    except Exception as e:
        print(f"[proxy] unhandled error: {e}", flush=True)
        return web.Response(status=502, text=f"Proxy error: {e}")

app = web.Application()
app.router.add_route("*", "/{tail:.*}", handle)

if __name__ == "__main__":
    print(f"[tool-proxy] {HOST}:{PORT} -> {BACKEND}", flush=True)
    web.run_app(app, host=HOST, port=PORT, print=lambda *a, **kw: None)
