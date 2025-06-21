import os
import torch
import traceback
import warnings
import base64
from io import BytesIO
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer
from runpod.serverless import start

# ──────────────────────────────────────────────────────────────────────────────
# ENV SETUP
# ──────────────────────────────────────────────────────────────────────────────
print("📦 Loading environment variables...")
load_dotenv(".env.local", override=True)

# Force Hugging Face cache paths
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

HF_TOKEN   = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
os.makedirs(MODEL_PATH, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD & CACHE MODEL
# ──────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        print("🔽 Downloading Juggernaut-XL…")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,
            cache_dir=os.environ["HF_HOME"]
        )
        pipe.save_pretrained(MODEL_PATH)
        pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
        if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2 is not None:
            pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))
        del pipe
    else:
        print("✅ Model already cached")

download_model_if_needed()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
print("🧠 Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
).to(DEVICE)

# Restore missing tokenizers
if pipe.tokenizer is None:
    pipe.tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_REPO,
        subfolder="tokenizer",
        use_auth_token=HF_TOKEN
    )
if not hasattr(pipe, "tokenizer_2") or pipe.tokenizer_2 is None:
    try:
        pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(
            MODEL_REPO,
            subfolder="tokenizer_2",
            use_auth_token=HF_TOKEN
        )
    except Exception:
        print("⚠️ No tokenizer_2 found. Continuing without it.")

# Try enabling xFormers
try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers enabled")
except Exception:
    print("⚠️ xFormers not available. Continuing...")

# ──────────────────────────────────────────────────────────────────────────────
# HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(event):
    try:
        print("🎯 Event received.")
        data = event.get("input", {})
        char = data.get("characterData", {})

        # Enriched photorealistic prompt
        prompt = (
            f"(photorealistic:1.4), ultra-detailed, 8K, studio lighting, realistic skin texture,\n"
            f"a portrait of a {char.get('gender','')} named {char.get('name','')}, {char.get('age','')} years old, "
            f"{char.get('race','')} race, {char.get('bodyType','')} body, "
            f"{char.get('hairColor','')} {char.get('hairStyle','')} hair, {char.get('eyeColor','')} eyes, "
            f"{char.get('boobSize','')} boobs, {char.get('buttSize','')} butt, "
            f"{char.get('personalityDescription','')}, background: {char.get('storylineBackground','')}, "
            f"setting: {char.get('setting','')}, relationship: {char.get('relationshipType','')}"
        )

        negative = data.get("negative_prompt", "cartoon, low res, painting, anime, sketch")
        guidance = float(data.get("guidance_scale", 10.0))
        steps    = int(data.get("steps", 60))

        print(f"🖼️ Generating with prompt: {prompt!r}")
        print(f"   negative_prompt: {negative!r} | steps: {steps} | scale: {guidance}")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=guidance,
            num_inference_steps=steps
        )
        image = result.images[0]

        # Encode into Base64 data-URL
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        print("✅ Image encoded and ready.")
        return {"image": data_url}

    except Exception as exc:
        print("❌ Error in handler:")
        traceback.print_exc()
        return {"error": str(exc), "trace": traceback.format_exc()}

# ──────────────────────────────────────────────────────────────────────────────
# RUN SERVERLESS
# ──────────────────────────────────────────────────────────────────────────────
print("🚀 Starting RunPod serverless handler...")
start({"handler": handler})
