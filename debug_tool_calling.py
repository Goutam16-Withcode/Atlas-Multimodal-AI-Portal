"""
Standalone tool-calling diagnostic. Run this directly:
    python debug_tool_calling.py

It sends one message that should obviously trigger generate_image_asset and
prints exactly what the model returned, so you can see whether the model is
calling the tool at all.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tools import ALL_TOOLS

MODEL = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
API_KEY = os.getenv("GROQ_API_KEY")

print(f"Testing model: {MODEL}")
print(f"API key present: {bool(API_KEY)}")
print(f"Tools bound: {[t.name for t in ALL_TOOLS]}\n")

llm = ChatGroq(model=MODEL, temperature=0.0, api_key=API_KEY).bind_tools(ALL_TOOLS)

messages = [
    SystemMessage(content="You are a helpful assistant with access to an image generation tool called generate_image_asset. Use it whenever asked to create an image."),
    HumanMessage(content="Generate an image of a robot welding on a factory floor."),
]

response = llm.invoke(messages)

print("=== RAW RESPONSE ===")
print(f"Content: {response.content!r}")
print(f"Tool calls: {response.tool_calls}")
print(f"Additional kwargs: {response.additional_kwargs}")

if not response.tool_calls:
    print("\n>>> DIAGNOSIS: The model did NOT call the tool. This is a model")
    print(">>> capability/reliability issue, not a bug in your app code.")
    print(">>> Try a different MODEL_NAME (e.g. llama-3.3-70b-versatile or")
    print(">>> llama-3.1-8b-instant) which have more reliable Groq tool-calling support.")
else:
    print("\n>>> DIAGNOSIS: Tool calling works fine at the model level.")
    print(">>> The bug is downstream — check ToolNode execution / graph wiring.")