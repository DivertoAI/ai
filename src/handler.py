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
# DOWNLOAD MODEL
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
        if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2:
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

if pipe.tokenizer is None:
    pipe.tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_REPO, subfolder="tokenizer", use_auth_token=HF_TOKEN
    )
if not hasattr(pipe, "tokenizer_2") or pipe.tokenizer_2 is None:
    try:
        pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(
            MODEL_REPO, subfolder="tokenizer_2", use_auth_token=HF_TOKEN
        )
    except Exception:
        print("⚠️ tokenizer_2 not found. Proceeding...")

try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()  # Optional memory optimization
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

        # Structured realism prompt
        prompt = (
            f"(photorealistic:1.4), ultra-detailed, cinematic lighting, HDR, realistic skin pores, "
            f"Canon EOS R5 85mm depth of field portrait of {char.get('gender','a person')} named {char.get('name','')}, "
            f"{char.get('age','')} years old, {char.get('race','')} race, {char.get('bodyType','')} physique, "
            f"{char.get('hairColor','')} {char.get('hairStyle','')} hair, {char.get('eyeColor','')} eyes, "
            f"{char.get('personalityDescription','')}, wearing natural clothing, "
            f"set in a {char.get('setting','simple setting')}, background: {char.get('storylineBackground','')}, "
            f"relationship status: {char.get('relationshipType','single')}"
        )

        negative = data.get("negative_prompt", "low quality, jpeg artifacts, cartoon, blurry, distorted, deformed face")
        guidance = float(data.get("guidance_scale", 9.5))
        steps    = int(data.get("steps", 75))

        print(f"🖼️ Prompt: {prompt}")
        print(f"⛔ Negative: {negative} | 🎚️ Steps: {steps} | 🎯 Guidance: {guidance}")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=guidance,
            num_inference_steps=steps
        )
        image = result.images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        print("✅ Image generation complete.")
        return {"image": data_url}

    except Exception as exc:
        print("❌ Exception:")
        traceback.print_exc()
        return {"error": str(exc), "trace": traceback.format_exc()}

# ──────────────────────────────────────────────────────────────────────────────
# START HANDLER
# ──────────────────────────────────────────────────────────────────────────────
print("🚀 Launching RunPod handler...")
start({"handler": handler})