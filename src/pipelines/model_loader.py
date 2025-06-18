import torch
from diffusers import StableDiffusionXLPipeline
from config.settings import settings
from loguru import logger

_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        logger.info(f"Loading {settings.model_repo} on {settings.device}")
        _pipe = StableDiffusionXLPipeline.from_pretrained(
            settings.model_repo,
            torch_dtype=torch.float16,
            cache_dir=settings.model_cache,
            use_auth_token=settings.hf_token,
        ).to(settings.device)
        try:
            _pipe.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled")
        except Exception:
            logger.warning("xFormers not available")
    return _pipe