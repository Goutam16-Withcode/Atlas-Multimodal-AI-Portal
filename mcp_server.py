"""
mcp_server.py — Expose Atlas's own tools as an MCP server (via FastMCP).

Running this standalone lets any MCP-compatible client (Claude Desktop,
Cursor, another LangGraph agent, etc.) discover and call Atlas's tools.

Usage:
    python mcp_server.py                       # defaults: host=127.0.0.1, port=9000
    python mcp_server.py --host 0.0.0.0 --port 9001

The server is exposed over HTTP using the Streamable-HTTP transport so it works
in cloud / Docker environments without relying on stdin/stdout.

If you only want to *consume* external MCP servers (not expose your own tools),
you don't need to run this file at all — just configure MCP_SERVERS_JSON in .env
and set MCP_ENABLED=true.
"""

import argparse
import logging

logger = logging.getLogger("mcp_server")

# ---------------------------------------------------------------------------
# Build the FastMCP server from the existing LangChain tools
# ---------------------------------------------------------------------------

def build_mcp_server():
    """Wrap every LangChain @tool in tools.py as a FastMCP tool and return the app."""
    try:
        from fastmcp import FastMCP  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "fastmcp is not installed. Run: pip install fastmcp"
        )

    from tools import ALL_TOOLS  # noqa: PLC0415

    mcp = FastMCP(
        name="Atlas Multimodal AI Portal MCP Server",
        instructions=(
            "This MCP server exposes the full tool suite of the Atlas industrial "
            "AI assistant: calculator, equipment status, knowledge-base search, "
            "image/video generation, support ticketing, web search, stock prices, "
            "weather, SQL queries, email, calendar, tasks, Slack, GitHub, and more."
        ),
    )

    for lc_tool in ALL_TOOLS:
        # Capture lc_tool in closure
        def _make_handler(t):
            async def handler(**kwargs):
                """Dynamically created FastMCP handler that delegates to the LangChain tool."""
                try:
                    # LangChain tools can be sync or async; invoke() handles both
                    result = t.invoke(kwargs)
                    return str(result)
                except Exception as exc:
                    logger.error(f"MCP tool '{t.name}' raised: {exc}")
                    return f"Error: {exc}"
            handler.__name__ = t.name
            handler.__doc__ = t.description
            return handler

        handler_fn = _make_handler(lc_tool)

        # Register with FastMCP
        mcp.tool(name=lc_tool.name, description=lc_tool.description)(handler_fn)
        logger.debug(f"Registered MCP tool: {lc_tool.name}")

    logger.info(f"FastMCP server built with {len(ALL_TOOLS)} tool(s).")
    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Atlas MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    mcp_app = build_mcp_server()
    # FastMCP exposes a standard ASGI app; run it with uvicorn
    uvicorn.run(
        mcp_app.http_app(),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
