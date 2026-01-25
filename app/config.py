import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    OPENAI_REALTIME_MODELS = {
        "gpt-realtime",
        "gpt-4o-realtime-preview",
    }
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(25 * 1024 * 1024)))
    SUPPORTED_MODELS = {
        "gemini-2.5-flash-lite": "gemini",
        "gpt-4o-mini": "openai",
        "gpt-realtime": "openai",
        "gpt-4o-realtime-preview": "openai",
    }


config = Config()
