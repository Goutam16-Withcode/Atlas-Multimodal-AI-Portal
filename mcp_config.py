"""
mcp_config.py — MCP (Model Context Protocol) client manager for Atlas.

This module:
  - Reads MCP server definitions from the environment (MCP_SERVERS_JSON)
    or from individual MCP_SERVER_<NAME>_* env vars.
  - Provides a lazy, app-lifetime MultiServerMCPClient instance.
  - Supports RUNTIME connect / disconnect of external MCP servers without
    restarting the process (via mcp_registry.py entries or raw definitions).
  - Exposes helpers to fetch live MCP tools and merge them with the
    chatbot's native LangChain tools so the agent sees a single unified
    tool list.

Environment variables
---------------------
MCP_SERVERS_JSON   JSON blob of server definitions (highest priority).
                   Example (SSE/HTTP server):
                     {
                       "filesystem": {
                         "transport": "stdio",
                         "command": "npx",
                         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                       },
                       "my_remote": {
                         "transport": "streamable_http",
                         "url": "http://localhost:9000/mcp"
                       }
                     }

MCP_ENABLED        "true" / "false"  (default: "false")
MCP_TIMEOUT        Per-tool timeout in seconds (default: 30)
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("mcp_config")

# ---------------------------------------------------------------------------
# Read static configuration from environment
# ---------------------------------------------------------------------------

MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "false").lower() == "true"
MCP_TIMEOUT: int = int(os.getenv("MCP_TIMEOUT", "30"))


def _load_server_definitions() -> Dict[str, Any]:
    """Load MCP server definitions from environment.

    Priority order:
      1. MCP_SERVERS_JSON  (full JSON blob)
      2. Individual MCP_SERVER_<NAME>_URL / MCP_SERVER_<NAME>_COMMAND vars
    """
    raw = os.getenv("MCP_SERVERS_JSON", "").strip()
    if raw:
        try:
            definitions = json.loads(raw)
            logger.info(f"Loaded {len(definitions)} MCP server(s) from MCP_SERVERS_JSON.")
            return definitions
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse MCP_SERVERS_JSON: {exc}")

    definitions: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith("MCP_SERVER_") and key.endswith("_URL"):
            server_name = key[len("MCP_SERVER_"):-len("_URL")].lower()
            definitions[server_name] = {"transport": "streamable_http", "url": value}
            logger.info(f"Registered MCP server '{server_name}' via HTTP at {value}")
        elif key.startswith("MCP_SERVER_") and key.endswith("_COMMAND"):
            server_name = key[len("MCP_SERVER_"):-len("_COMMAND")].lower()
            parts = value.split()
            args_key = f"MCP_SERVER_{server_name.upper()}_ARGS"
            extra_args = os.getenv(args_key, "").split() if os.getenv(args_key) else parts[1:]
            definitions[server_name] = {
                "transport": "stdio",
                "command": parts[0] if parts else value,
                "args": extra_args,
            }
            logger.info(f"Registered MCP server '{server_name}' via stdio: {value}")

    return definitions


# ---------------------------------------------------------------------------
# Runtime-mutable server registry
# ---------------------------------------------------------------------------

# Starts from env / JSON config; keys can be added/removed at runtime.
SERVER_DEFINITIONS: Dict[str, Any] = _load_server_definitions()

# Tracks which server_ids were connected at runtime (for disconnect)
_runtime_connected: Dict[str, Any] = {}


def get_active_server_definitions() -> Dict[str, Any]:
    """Return the merged set of env-loaded + runtime-connected server definitions."""
    merged = dict(SERVER_DEFINITIONS)
    merged.update(_runtime_connected)
    return merged


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_mcp_client = None           # MultiServerMCPClient instance (lazy)
_mcp_tools_cache: List = []  # Last fetched tool list


async def _make_client() -> Optional[Any]:
    """Instantiate (or re-instantiate) the MultiServerMCPClient."""
    global _mcp_client

    active = get_active_server_definitions()
    if not active:
        logger.warning("No MCP server definitions found. MCP tools disabled.")
        return None

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: PLC0415
        _mcp_client = MultiServerMCPClient(active)
        logger.info(
            f"MultiServerMCPClient (re)initialized with servers: {list(active.keys())}"
        )
        return _mcp_client
    except ImportError:
        logger.error(
            "langchain-mcp-adapters is not installed. "
            "Run: pip install langchain-mcp-adapters"
        )
        return None
    except Exception as exc:
        logger.error(f"Failed to initialize MCP client: {exc}")
        return None


async def get_mcp_client():
    """Return (and lazily initialize) the global MultiServerMCPClient.

    Returns None when MCP_ENABLED is false and no runtime servers are active.
    """
    global _mcp_client

    has_runtime = bool(_runtime_connected)

    if not MCP_ENABLED and not has_runtime:
        return None

    if _mcp_client is None:
        _mcp_client = await _make_client()

    return _mcp_client


async def get_mcp_tools(refresh: bool = False) -> List:
    """Fetch LangChain-compatible tools from all connected MCP servers.

    Results are cached for the lifetime of the process unless refresh=True.
    Returns an empty list when MCP is disabled or client init fails.
    """
    global _mcp_tools_cache

    has_runtime = bool(_runtime_connected)
    if not MCP_ENABLED and not has_runtime:
        return []

    if _mcp_tools_cache and not refresh:
        return _mcp_tools_cache

    client = await get_mcp_client()
    if client is None:
        return []

    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=MCP_TIMEOUT)
        _mcp_tools_cache = tools
        logger.info(
            f"Fetched {len(tools)} tool(s) from MCP servers: {[t.name for t in tools]}"
        )
        return tools
    except asyncio.TimeoutError:
        logger.error(f"MCP get_tools() timed out after {MCP_TIMEOUT}s.")
        return _mcp_tools_cache
    except Exception as exc:
        logger.error(f"Failed to fetch MCP tools: {exc}")
        return _mcp_tools_cache


async def get_all_tools(native_tools: List) -> List:
    """Merge native LangChain tools with live MCP tools.

    MCP tools are appended after native tools. Duplicate names are resolved
    in favour of the native tool (so built-in behaviour is never overridden).
    """
    mcp_tools = await get_mcp_tools()
    if not mcp_tools:
        return native_tools

    native_names = {t.name for t in native_tools}
    merged = list(native_tools)
    added, skipped = [], []
    for t in mcp_tools:
        if t.name in native_names:
            skipped.append(t.name)
        else:
            merged.append(t)
            added.append(t.name)

    if added:
        logger.info(f"MCP tools added to agent: {added}")
    if skipped:
        logger.debug(f"MCP tools skipped (name collision with native): {skipped}")

    return merged


# ---------------------------------------------------------------------------
# Runtime connect / disconnect
# ---------------------------------------------------------------------------

async def connect_server(server_id: str, definition: Dict[str, Any]) -> List:
    """Dynamically connect a new MCP server and refresh the tool cache.

    Args:
        server_id:  Unique key for this server (e.g. "google_drive").
        definition: MultiServerMCPClient-compatible dict.

    Returns:
        Updated full MCP tool list.
    """
    global _mcp_client, _mcp_tools_cache

    _runtime_connected[server_id] = definition
    logger.info(f"Connecting MCP server '{server_id}': {definition.get('transport')} transport")

    # Re-create client with updated server list
    _mcp_client = await _make_client()
    # Force tool refresh
    _mcp_tools_cache = []
    return await get_mcp_tools(refresh=True)


async def disconnect_server(server_id: str) -> List:
    """Disconnect a runtime-connected MCP server and refresh the tool cache.

    Also works for env-loaded servers (removes from SERVER_DEFINITIONS too).

    Returns:
        Updated full MCP tool list.
    """
    global _mcp_client, _mcp_tools_cache

    removed = False
    if server_id in _runtime_connected:
        del _runtime_connected[server_id]
        removed = True
    if server_id in SERVER_DEFINITIONS:
        del SERVER_DEFINITIONS[server_id]
        removed = True

    if not removed:
        raise KeyError(f"Server '{server_id}' is not currently connected.")

    logger.info(f"Disconnected MCP server '{server_id}'.")

    # Re-create client or set to None if no servers remain
    active = get_active_server_definitions()
    if active:
        _mcp_client = await _make_client()
    else:
        _mcp_client = None
        logger.info("No MCP servers remain — MCP client cleared.")

    _mcp_tools_cache = []
    return await get_mcp_tools(refresh=True)


def get_connected_servers() -> Dict[str, Any]:
    """Return all currently connected servers (env + runtime), without secrets."""
    active = get_active_server_definitions()
    safe = {}
    for sid, defn in active.items():
        safe_defn = {k: v for k, v in defn.items() if k not in ("env", "headers")}
        safe[sid] = safe_defn
    return safe


def get_server_status() -> Dict[str, Any]:
    """Return a serialisable summary of MCP configuration (for /mcp/status)."""
    active = get_active_server_definitions()
    return {
        "enabled": MCP_ENABLED,
        "servers_active": len(active),
        "server_ids": list(active.keys()),
        "tools_cached": len(_mcp_tools_cache),
        "tool_names": [t.name for t in _mcp_tools_cache],
        "timeout_seconds": MCP_TIMEOUT,
    }
