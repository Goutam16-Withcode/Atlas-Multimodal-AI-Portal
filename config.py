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
    BACKUP_GROQ_API_KEY: str = os.getenv("BACKUP_GROQ_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Memory / checkpointing
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "chatbot_memory.db")

    # Assets & exports storage
    ASSET_STORAGE_DIR: str = os.getenv("ASSET_STORAGE_DIR", os.path.join("static", "generated"))
    EXPORT_STORAGE_DIR: str = os.getenv("EXPORT_STORAGE_DIR", os.path.join("static", "exports"))

    # Image / Video Generation
    IMAGE_GEN_BASE_URL: str = os.getenv("IMAGE_GEN_BASE_URL", "https://image.pollinations.ai/prompt")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
    REPLICATE_VIDEO_MODEL_VERSION: str = os.getenv("REPLICATE_VIDEO_MODEL_VERSION", "stability-ai/stable-video-diffusion:3f04b56b85b77a65c8a45e99a65f02f69a56fba1051515151515151515151515")
    VIDEO_GEN_MAX_POLL_SECONDS: int = int(os.getenv("VIDEO_GEN_MAX_POLL_SECONDS", "300"))
    VIDEO_GEN_POLL_INTERVAL_SECONDS: int = int(os.getenv("VIDEO_GEN_POLL_INTERVAL_SECONDS", "10"))

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

    @property
    def groq_api_keys(self) -> list:
        keys = []
        if self.GROQ_API_KEY:
            keys.append(self.GROQ_API_KEY.strip().strip('"'))
        if self.BACKUP_GROQ_API_KEY:
            keys.append(self.BACKUP_GROQ_API_KEY.strip().strip('"'))
        
        env_keys = os.getenv("GROQ_API_KEYS", "")
        if env_keys:
            for k in env_keys.split(","):
                k_clean = k.strip().strip('"')
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)
        return keys


settings = Settings()