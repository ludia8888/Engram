"""Remote MCP transport: SSE and StreamableHTTP app creation tests."""
from __future__ import annotations

import pytest

from engram.mcp_server import build_mcp_server, _VALID_TRANSPORTS, _resolve_transport


def test_build_mcp_server_registers_all_tools():
    server = build_mcp_server()
    tool_names = sorted(server._tool_manager._tools.keys())
    assert tool_names == [
        "engram_append",
        "engram_flush",
        "engram_get",
        "engram_get_relations",
        "engram_history",
        "engram_recall",
        "engram_search",
        "engram_turn",
    ]


def test_build_mcp_server_with_custom_host_port():
    server = build_mcp_server(host="0.0.0.0", port=9090)
    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 9090
    assert len(server._tool_manager._tools) == 8


def test_build_mcp_server_default_host_port():
    server = build_mcp_server()
    # Default FastMCP settings (no host/port override)
    assert server.settings.host == "127.0.0.1"
    assert server.settings.port == 8000


def test_sse_app_is_starlette_application():
    server = build_mcp_server(host="0.0.0.0", port=8080)
    app = server.sse_app()
    # Starlette app has routes attribute
    assert hasattr(app, "routes")


def test_streamable_http_app_is_starlette_application():
    server = build_mcp_server(host="0.0.0.0", port=8080)
    app = server.streamable_http_app()
    assert hasattr(app, "routes")


def test_resolve_transport_defaults_to_stdio(monkeypatch):
    monkeypatch.delenv("ENGRAM_MCP_TRANSPORT", raising=False)
    assert _resolve_transport(default="stdio") == "stdio"


def test_resolve_transport_reads_env(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_TRANSPORT", "sse")
    assert _resolve_transport(default="stdio") == "sse"


def test_resolve_transport_rejects_invalid(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_TRANSPORT", "websocket")
    with pytest.raises(ValueError, match="websocket"):
        _resolve_transport(default="stdio")


def test_resolve_transport_accepts_all_valid(monkeypatch):
    for transport in _VALID_TRANSPORTS:
        monkeypatch.setenv("ENGRAM_MCP_TRANSPORT", transport)
        assert _resolve_transport() == transport


def test_multiple_servers_are_independent():
    """Two build_mcp_server() calls produce independent instances."""
    server_a = build_mcp_server(host="127.0.0.1", port=8001)
    server_b = build_mcp_server(host="0.0.0.0", port=8002)
    assert server_a is not server_b
    assert server_a.settings.port == 8001
    assert server_b.settings.port == 8002
    assert len(server_a._tool_manager._tools) == 8
    assert len(server_b._tool_manager._tools) == 8
