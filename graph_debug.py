"""
Tests the FULL graph (same path /chat and /chat/stream use), not just the
tool in isolation. This is the closest thing to "what does the user actually
see in the chat window."
Run: python debug_full_graph.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from graph import chatbot

config = {"configurable": {"thread_id": "debug-thread-image", "username": "debug_user", "language": "en"}}

print("Invoking full graph with an image request...\n")
result = chatbot.invoke(
    {"messages": [HumanMessage(content="Generate an image of a robot welding on a factory floor.")]},
    config=config,
)

final_message = result["messages"][-1]
print("=== FINAL MESSAGE CONTENT (what /chat would return as `reply`) ===")
print(repr(final_message.content))
print("\n=== Does it contain an <img> tag? ===")
print("<img" in final_message.content)

print("\n=== ALL messages in final state (to see every step) ===")
for i, m in enumerate(result["messages"]):
    print(f"\n--- [{i}] {type(m).__name__} ---")
    print(repr(m.content)[:500])
    if hasattr(m, "tool_calls") and m.tool_calls:
        print(f"tool_calls: {m.tool_calls}")