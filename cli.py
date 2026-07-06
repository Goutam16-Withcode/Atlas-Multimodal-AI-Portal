"""
Terminal chat client for local testing.

Usage:
    python cli.py --thread plant-a-shift-1
"""
import argparse
from langchain_core.messages import HumanMessage
from graph import chatbot
from logger import get_logger

logger = get_logger("cli")


def main():
    parser = argparse.ArgumentParser(description="Industrial Support Chatbot CLI")
    parser.add_argument("--thread", default="default-session",
                         help="Conversation/thread ID (persists across restarts)")
    args = parser.parse_args()

    config = {
        "configurable": {"thread_id": args.thread},
        "metadata": {
            "thread_id": args.thread,
            "interface": "cli"
        }
    }
    print(f"Atlas Industrial Support Bot — thread '{args.thread}'. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        reply = result["messages"][-1]
        print(f"Atlas: {reply.content}\n")


if __name__ == "__main__":
    main()