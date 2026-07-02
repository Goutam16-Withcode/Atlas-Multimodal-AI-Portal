"""
Assembles the full LangGraph state machine:

        START
          |
   classify_intent
          |
   (escalation?) --yes--> escalation_check --> compress_history
          |no
   compress_history
          |
      agent_node <-------------------+
          |                          |
   (tool calls?) --yes--> tool_node --+
          |no
         END

Persistence uses SqliteSaver so conversations survive process restarts —
required for any real deployment (vs. the original InMemorySaver, which
loses everything when the process exits).
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from state import ChatState
from nodes import (
    classify_intent,
    compress_history,
    agent_node,
    tool_node,
    escalation_check,
    route_after_agent,
    route_after_intent,
)
from config import settings
from logger import get_logger

logger = get_logger(__name__)


def build_graph(checkpointer=None):
    graph = StateGraph(ChatState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("escalation_check", escalation_check)
    graph.add_node("compress_history", compress_history)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"escalation_check": "escalation_check", "compress_history": "compress_history"},
    )
    graph.add_edge("escalation_check", "compress_history")
    graph.add_edge("compress_history", "agent_node")

    graph.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent_node")

    if checkpointer is None:
        conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        checkpointer = SqliteSaver(conn)
        logger.info(f"Using SQLite persistence at {settings.SQLITE_DB_PATH}")

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# Default compiled chatbot instance
chatbot = build_graph()