from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipelines.model_loader import get_pipeline
from utils.logger import logger
import uuid
import os

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    steps: int = 30
    guidance_scale: float = 7.5

@app.post("/generate")
async def generate(req: GenerationRequest):
    pipe = get_pipeline()
    try:
        logger.info("Generating image for prompt: %s", req.prompt)
        image = pipe(
            req.prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.steps
        ).images[0]
        filename = f"{uuid.uuid4()}.png"
        out_path = os.path.join("public", filename)
        os.makedirs("public", exist_ok=True)
        image.save(out_path)
        return {"url": f"/{filename}"}
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))