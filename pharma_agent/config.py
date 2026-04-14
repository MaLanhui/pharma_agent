from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "")
    pubmed_tool: str = os.getenv("PUBMED_TOOL", "pharma-agent-demo")
    local_embedding_mode: str = os.getenv("LOCAL_EMBEDDING_MODE", "hash").lower()
    swissadme_timeout: int = int(os.getenv("SWISSADME_TIMEOUT", "90"))
    swissadme_cache_enabled: bool = os.getenv("SWISSADME_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
    swissadme_delay_seconds: float = float(os.getenv("SWISSADME_DELAY_SECONDS", "3.0"))
    protox3_timeout: int = int(os.getenv("PROTOX3_TIMEOUT", "120"))
    protox3_cache_enabled: bool = os.getenv("PROTOX3_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
    protox3_delay_seconds: float = float(os.getenv("PROTOX3_DELAY_SECONDS", "6.0"))
    protox3_refine_top_n: int = int(os.getenv("PROTOX3_REFINE_TOP_N", "2"))
    protox3_max_retries: int = int(os.getenv("PROTOX3_MAX_RETRIES", "3"))
    protox3_retry_backoff_seconds: float = float(os.getenv("PROTOX3_RETRY_BACKOFF_SECONDS", "8.0"))


settings = Settings()
SWISSADME_CACHE_PATH = DATA_DIR / "swissadme_cache.json"
PROTOX3_CACHE_PATH = DATA_DIR / "protox3_cache.json"
