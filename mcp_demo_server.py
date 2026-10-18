#!/usr/bin/env python3
"""Tiny stdio MCP server used to exercise the Workbench's MCP client.

Implements JSON-RPC 2.0: initialize, notifications/initialized, tools/list,
tools/call. Tools: now(), echo(text), rand(n), disk_free().
"""
import datetime
import json
import random
import shutil
import sys

TOOLS = [
    {
        "name": "now",
        "description": "Current local date and time (ISO 8601).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "rand",
        "description": "Random integer in [0, n).",
        "inputSchema": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        },
    },
    {
        "name": "disk_free",
        "description": "Free disk space on the boot volume in GB.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call(name: str, args: dict) -> str:
    if name == "now":
        return datetime.datetime.now().isoformat(timespec="seconds")
    if name == "echo":
        return str(args.get("text", ""))
    if name == "rand":
        return str(random.randrange(int(args.get("n", 100) or 100)))
    if name == "disk_free":
        return f"{shutil.disk_usage('/').free / 2**30:.1f} GB free"
    raise ValueError(f"unknown tool {name}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in msg:          # notification
            continue
        method, params, rid = msg.get("method"), msg.get("params") or {}, msg["id"]
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "demo", "version": "1.0"},
                }
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": call(params["name"], params.get("arguments") or {})}]}
            else:
                raise ValueError(f"unsupported method {method}")
            out = {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as e:
            out = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
