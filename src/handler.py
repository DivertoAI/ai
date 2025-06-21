import os
import torch
import traceback
import warnings
import base64
import shutil
from io import BytesIO
from dotenv import load_dotenv
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTokenizer
from runpod.serverless import start
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
#  ENV SETUP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(
    dotenv_path=os.getenv("DOTENV_PATH", ".env.local"),
    override=True
)

# force HF cache
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
ios.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface"
ios.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface"
for k in ["HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_TELEMETRY"]:
    os.environ[k] = "1"

HF_TOKEN   = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
os.makedirs(MODEL_PATH, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN,
            cache_dir=os.environ["HF_HOME"]
        )
        pipe.save_pretrained(MODEL_PATH)
        pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
        if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2:
            pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))
        del pipe

        print("🔽 Model downloaded & cached.")
    else:
        print("✅ Model cache found.")

download_model_if_needed()

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
print("🧠 Loading SDXL pipeline with DPMSolverMultistepScheduler…")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
).to(DEVICE)

# restore tokenizers
if pipe.tokenizer is None:
    pipe.tokenizer = CLIPTokenizer.from_pretrained(MODEL_REPO, subfolder="tokenizer", use_auth_token=HF_TOKEN)
if not getattr(pipe, "tokenizer_2", None):
    try:
        pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL_REPO, subfolder="tokenizer_2", use_auth_token=HF_TOKEN)
    except:
        pass

# memory efficient attention
try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers enabled.")
except:
    print("⚠️ xFormers not available.")

# ──────────────────────────────────────────────────────────────────────────────
#  HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(event):
    try:
        data = event.get("input", {})
        char = data.get("characterData", {})

        # build ultra-photorealistic prompt
        prompt = (
            f"ultra realistic, photorealistic portrait of a {char.get('gender','')} named {char.get('name','')}, "
            f"{char.get('age','')} years old, {char.get('race','')} race, {char.get('bodyType','')} body, "
            f"{char.get('hairColor','')} {char.get('hairStyle','')} hair, {char.get('eyeColor','')} eyes, "
            f"{char.get('boobSize','')} boobs, {char.get('buttSize','')} butt, sub-surface scattering skin, "
            f"8k resolution, studio lighting, film grain, cinematic depth of field, extremely detailed skin pores"
        )

        negative = data.get("negative_prompt", "cartoon, anime, sketch, lowres, deformed, blurry")
        steps = int(data.get("steps", 50))
        guidance = float(data.get("guidance_scale", 9.0))

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        img = out.images[0]

        # optional upscaling with Real-ESRGAN (if installed)
        try:
            from realesrgan import RealESRGAN
            model = RealESRGAN(DEVICE, scale=2)
            model.load_weights("RealESRGAN_x2.pth")
            img = model.predict(img)
        except:
            pass

        # encode as Base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {"image": f"data:image/png;base64,{b64}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}

# ──────────────────────────────────────────────────────────────────────────────
#  RUN SERVERLESS
# ──────────────────────────────────────────────────────────────────────────────
print("🚀 Starting RunPod handler…")
start({"handler": handler})