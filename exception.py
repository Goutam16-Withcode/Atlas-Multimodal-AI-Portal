"""
Tests generate_image_asset directly, bypassing the LLM entirely.
Run: python debug_tool_execution.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from tools import generate_image_asset

print("Calling generate_image_asset directly...\n")
try:
    result = generate_image_asset.invoke({"prompt": "a robot welding on a factory floor"})
    print("=== TOOL RESULT ===")
    print(result)
except Exception as e:
    import traceback
    print("=== TOOL RAISED AN EXCEPTION ===")
    traceback.print_exc()