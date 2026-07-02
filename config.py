"""
Centralized configuration for the Industrial Support Chatbot.
All tunables live here so the rest of the codebase never hardcodes values.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Memory / checkpointing
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "chatbot_memory.db")

    # Conversation management
    MAX_MESSAGES_BEFORE_SUMMARY: int = int(os.getenv("MAX_MESSAGES_BEFORE_SUMMARY", "12"))
    KEEP_LAST_N_AFTER_SUMMARY: int = int(os.getenv("KEEP_LAST_N_AFTER_SUMMARY", "4"))

    # Retry / resilience
    MAX_LLM_RETRIES: int = int(os.getenv("MAX_LLM_RETRIES", "3"))
    RETRY_BACKOFF_SECONDS: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

    # Escalation thresholds
    ESCALATE_ON_KEYWORDS = [
        "not working", "urgent", "critical", "down", "failure",
        "leak", "fire", "smoke", "injury", "unsafe", "emergency",
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()