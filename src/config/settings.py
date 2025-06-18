import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    hf_token: str
    model_repo: str = "RunDiffusion/Juggernaut-XL-v8"
    model_cache: str = "/runpod-volume/huggingface"
    device: str = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

    class Config:
        env_file = ".env"

settings = Settings()