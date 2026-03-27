"""Application configuration via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """CCTV Intelligence Platform configuration.

    Reads values from environment variables or a .env file.
    """

    database_url: str = "postgresql+asyncpg://cctv_user:changeme@localhost:5432/cctv_db"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_path: str | None = None  # Use for local storage if Docker is missing
    redis_url: str = "redis://localhost:6379/0"
    anthropic_api_key: str = "your_key_here"
    clips_dir: str = "./clips"
    videos_dir: str = "./videos"
    yolo_model: str = "yolov8m.pt"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    sample_fps: float = 2.0
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this everywhere
settings = Settings()
