"""
mcp_registry.py — Catalogue of well-known external MCP servers.

Each entry describes:
  - id           : Unique slug used as the key in the MCP client config
  - name         : Human-readable display name
  - description  : What it provides
  - category     : Grouping for the UI
  - transport    : "stdio" | "streamable_http" | "sse"
  - command/args : For stdio servers (typically npx-based)
  - url_template : For HTTP servers (may include {param} placeholders)
  - auth_env     : List of env var names required for authentication
  - config_params: Extra parameters the user must supply at connect-time
  - docs_url     : Where to find setup instructions
  - icon         : Emoji icon for the UI

Users add their credentials via environment variables — the chatbot
never stores raw secrets in the database.
"""

from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Dict[str, Any]] = {

    # ── Google ──────────────────────────────────────────────────────────────
    "google_drive": {
        "id": "google_drive",
        "name": "Google Drive",
        "description": "Read, search, and list files in Google Drive. "
                       "Supports Docs, Sheets, PDFs, and more.",
        "category": "Productivity",
        "icon": "📁",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gdrive"],
        "auth_env": ["GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/gdrive",
        "env_passthrough": ["GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN"],
    },

    "gmail": {
        "id": "gmail",
        "name": "Gmail",
        "description": "Read, search, send, and manage Gmail messages and threads.",
        "category": "Communication",
        "icon": "📧",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gmail"],
        "auth_env": ["GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/gmail",
        "env_passthrough": ["GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"],
    },

    "google_calendar": {
        "id": "google_calendar",
        "name": "Google Calendar",
        "description": "Read and create Google Calendar events, list upcoming meetings.",
        "category": "Productivity",
        "icon": "📅",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-google-calendar"],
        "auth_env": ["GOOGLE_CALENDAR_CLIENT_ID", "GOOGLE_CALENDAR_CLIENT_SECRET",
                     "GOOGLE_CALENDAR_REFRESH_TOKEN"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers",
        "env_passthrough": ["GOOGLE_CALENDAR_CLIENT_ID", "GOOGLE_CALENDAR_CLIENT_SECRET",
                            "GOOGLE_CALENDAR_REFRESH_TOKEN"],
    },

    # ── Developer tools ─────────────────────────────────────────────────────
    "github": {
        "id": "github",
        "name": "GitHub",
        "description": "Search repos, read files, create/update issues and PRs, "
                       "list branches, manage code reviews.",
        "category": "Developer",
        "icon": "🐙",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "auth_env": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/github",
        "env_passthrough": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    },

    "gitlab": {
        "id": "gitlab",
        "name": "GitLab",
        "description": "Interact with GitLab repos, issues, MRs, pipelines, and wikis.",
        "category": "Developer",
        "icon": "🦊",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gitlab"],
        "auth_env": ["GITLAB_PERSONAL_ACCESS_TOKEN", "GITLAB_API_URL"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/gitlab",
        "env_passthrough": ["GITLAB_PERSONAL_ACCESS_TOKEN", "GITLAB_API_URL"],
    },

    # ── File system & databases ─────────────────────────────────────────────
    "filesystem": {
        "id": "filesystem",
        "name": "Local Filesystem",
        "description": "Read, write, search, and list files in a specified local directory.",
        "category": "Files",
        "icon": "🗂️",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "{allowed_path}"],
        "auth_env": [],
        "config_params": [
            {
                "key": "allowed_path",
                "label": "Allowed Directory Path",
                "placeholder": "C:/Users/YourName/Documents",
                "required": True,
            }
        ],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        "env_passthrough": [],
    },

    "postgres": {
        "id": "postgres",
        "name": "PostgreSQL",
        "description": "Query a PostgreSQL database, explore schemas, run safe read-only SQL.",
        "category": "Database",
        "icon": "🐘",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", "{connection_string}"],
        "auth_env": ["PG_CONNECTION_STRING"],
        "config_params": [
            {
                "key": "connection_string",
                "label": "PostgreSQL Connection String",
                "placeholder": "postgresql://user:pass@localhost:5432/mydb",
                "required": True,
                "secret": True,
            }
        ],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/postgres",
        "env_passthrough": [],
    },

    # ── Communication & collaboration ───────────────────────────────────────
    "slack": {
        "id": "slack",
        "name": "Slack",
        "description": "Post messages, read channels, search Slack workspace history, "
                       "manage threads and reactions.",
        "category": "Communication",
        "icon": "💬",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "auth_env": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/slack",
        "env_passthrough": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    },

    "notion": {
        "id": "notion",
        "name": "Notion",
        "description": "Read and create Notion pages and databases. "
                       "Search workspace content, update properties.",
        "category": "Productivity",
        "icon": "📝",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-notion"],
        "auth_env": ["NOTION_API_KEY"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/notion",
        "env_passthrough": ["NOTION_API_KEY"],
    },

    # ── Search & web ─────────────────────────────────────────────────────────
    "brave_search": {
        "id": "brave_search",
        "name": "Brave Search",
        "description": "Real-time web search powered by Brave Search API — "
                       "privacy-respecting, no tracking.",
        "category": "Search",
        "icon": "🦁",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "auth_env": ["BRAVE_API_KEY"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search",
        "env_passthrough": ["BRAVE_API_KEY"],
    },

    "puppeteer": {
        "id": "puppeteer",
        "name": "Browser (Puppeteer)",
        "description": "Control a real headless browser — navigate URLs, click, "
                       "fill forms, take screenshots, extract page content.",
        "category": "Browser",
        "icon": "🌐",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "auth_env": [],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer",
        "env_passthrough": [],
    },

    # ── Cloud & storage ──────────────────────────────────────────────────────
    "aws_kb_retrieval": {
        "id": "aws_kb_retrieval",
        "name": "AWS Knowledge Base (Bedrock)",
        "description": "Query Amazon Bedrock Knowledge Bases using hybrid RAG retrieval.",
        "category": "Cloud",
        "icon": "☁️",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-aws-kb-retrieval-mcp-server"],
        "auth_env": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        "config_params": [],
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/aws-kb-retrieval-mcp-server",
        "env_passthrough": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    },

    # ── Custom / Self-hosted ─────────────────────────────────────────────────
    "custom_http": {
        "id": "custom_http",
        "name": "Custom HTTP MCP Server",
        "description": "Connect to any remote MCP server via Streamable HTTP transport. "
                       "Supports self-hosted FastMCP servers, corporate MCP gateways, etc.",
        "category": "Custom",
        "icon": "🔌",
        "transport": "streamable_http",
        "url_template": "{url}",
        "auth_env": [],
        "config_params": [
            {
                "key": "url",
                "label": "MCP Server URL",
                "placeholder": "http://localhost:9001/mcp",
                "required": True,
            },
            {
                "key": "api_key",
                "label": "API Key (if required)",
                "placeholder": "sk-...",
                "required": False,
                "secret": True,
            },
        ],
        "docs_url": "https://modelcontextprotocol.io/specification",
        "env_passthrough": [],
    },

    "custom_stdio": {
        "id": "custom_stdio",
        "name": "Custom stdio MCP Server",
        "description": "Launch any local MCP server process via stdio. "
                       "Works with Python (fastmcp), Node.js, or any MCP-compatible binary.",
        "category": "Custom",
        "icon": "⚙️",
        "transport": "stdio",
        "command": "{command}",
        "args_template": "{args}",
        "auth_env": [],
        "config_params": [
            {
                "key": "command",
                "label": "Executable Command",
                "placeholder": "python",
                "required": True,
            },
            {
                "key": "args",
                "label": "Arguments (space-separated)",
                "placeholder": "my_mcp_server.py",
                "required": False,
            },
        ],
        "docs_url": "https://modelcontextprotocol.io/specification",
        "env_passthrough": [],
    },

    # ── Self (Atlas back onto itself) ────────────────────────────────────────
    "atlas_self": {
        "id": "atlas_self",
        "name": "Atlas MCP Server (self)",
        "description": "Connect this Atlas instance to its own MCP server so external "
                       "MCP clients (Claude Desktop, Cursor, etc.) can call Atlas tools. "
                       "Also lets you test the MCP server locally.",
        "category": "Built-in",
        "icon": "🤖",
        "transport": "streamable_http",
        "url_template": "http://{host}:{port}/mcp",
        "auth_env": [],
        "config_params": [
            {
                "key": "host",
                "label": "MCP Server Host",
                "placeholder": "127.0.0.1",
                "required": False,
            },
            {
                "key": "port",
                "label": "MCP Server Port",
                "placeholder": "9000",
                "required": False,
            },
        ],
        "docs_url": "https://github.com/Goutam16-Withcode/Multitask_chatbot",
        "env_passthrough": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_registry() -> List[Dict[str, Any]]:
    """Return the full registry as a list, sorted by category then name."""
    entries = []
    for entry in REGISTRY.values():
        entries.append({k: v for k, v in entry.items() if k != "env_passthrough"})
    return sorted(entries, key=lambda e: (e["category"], e["name"]))


def get_entry(server_id: str) -> Optional[Dict[str, Any]]:
    """Return the registry entry for a given server ID, or None."""
    return REGISTRY.get(server_id)


def build_server_definition(
    server_id: str,
    user_params: Dict[str, str],
    env_override: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a MultiServerMCPClient-compatible definition from a registry entry + user params.

    Args:
        server_id:   Registry key (e.g. "google_drive", "custom_http").
        user_params: User-supplied config_params values (keyed by param 'key').
        env_override: Optional extra environment variables to inject into the subprocess.

    Returns:
        A dict suitable for MultiServerMCPClient, or None if the entry is missing.
    """
    import os  # noqa: PLC0415

    entry = REGISTRY.get(server_id)
    if not entry:
        return None

    transport = entry["transport"]
    env_vars = dict(env_override or {})

    # Collect env vars defined in env_passthrough (reads from the process environment)
    for var in entry.get("env_passthrough", []):
        val = os.getenv(var, "")
        if val:
            env_vars[var] = val

    if transport == "stdio":
        command = entry.get("command", "")
        args = list(entry.get("args", []))

        # Substitute {param} placeholders in args
        substituted_args = []
        for arg in args:
            if arg.startswith("{") and arg.endswith("}"):
                key = arg[1:-1]
                substituted_args.append(user_params.get(key, ""))
            else:
                substituted_args.append(arg)

        # Handle dynamic command
        if command.startswith("{") and command.endswith("}"):
            key = command[1:-1]
            command = user_params.get(key, "")

        # Handle args_template (custom_stdio: space-separated args string)
        if "args_template" in entry:
            raw_args = user_params.get("args", "")
            substituted_args = raw_args.split() if raw_args else []

        definition: Dict[str, Any] = {
            "transport": "stdio",
            "command": command,
            "args": substituted_args,
        }
        if env_vars:
            definition["env"] = env_vars

    elif transport in ("streamable_http", "sse"):
        url_template = entry.get("url_template", "{url}")

        # Substitute placeholders in URL
        url = url_template
        for key, val in user_params.items():
            url = url.replace("{" + key + "}", val)

        # Defaults for atlas_self
        url = url.replace("{host}", user_params.get("host", "127.0.0.1"))
        url = url.replace("{port}", user_params.get("port", "9000"))
        url = url.replace("{url}", user_params.get("url", ""))

        definition = {
            "transport": transport,
            "url": url,
        }

        # Optional bearer token for custom_http
        if "api_key" in user_params and user_params["api_key"]:
            definition["headers"] = {"Authorization": f"Bearer {user_params['api_key']}"}

    else:
        return None

    return definition


def list_categories() -> List[str]:
    """Return sorted list of unique categories in the registry."""
    return sorted({e["category"] for e in REGISTRY.values()})
