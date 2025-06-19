import os
import torch
import traceback
import warnings
import shutil
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer
from runpod.serverless import start

# ──────────────────────────────────────────────────────────────────────────────
#  ENV SETUP
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
#  DOWNLOAD & CACHE MODEL
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
#  LOAD PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
print("🧠 Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
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
    except Exception as e:
        print("⚠️ No tokenizer_2 found. Continuing without it.")

# Try enabling xFormers
try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers enabled")
except Exception:
    print("⚠️ xFormers not available. Continuing...")

# ──────────────────────────────────────────────────────────────────────────────
#  HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(event):
    try:
        print("🎯 Event received.")
        data = event.get("input", {})
        char = data.get("characterData", {})

        prompt = (
            f"a portrait of a {char.get('gender','')} named {char.get('name','')}, "
            f"{char.get('age','')} years old, {char.get('race','')} race, "
            f"{char.get('bodyType','')} body, {char.get('hairColor','')} {char.get('hairStyle','')} hair, "
            f"{char.get('eyeColor','')} eyes, {char.get('boobSize','')} boobs, {char.get('buttSize','')} butt, "
            f"{char.get('personalityDescription','')}, background: {char.get('storylineBackground','')}, "
            f"setting: {char.get('setting','')}, relationship: {char.get('relationshipType','')}, cinematic lighting, high quality"
        )

        guidance = float(data.get("guidance_scale", 7.5))
        steps    = int(data.get("steps", 30))

        print(f"🖼️ Generating: {prompt!r} | steps: {steps} | scale: {guidance}")

        result = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps)
        image = result.images[0]

        tmp = "/tmp/output.png"
        public = "/runpod-volume/public/output.png"
        os.makedirs(os.path.dirname(public), exist_ok=True)
        image.save(tmp)
        shutil.copy(tmp, public)

        print("✅ Image saved successfully.")
        return {"image_paths": ["/output.png"]}

    except Exception as exc:
        print("❌ Error in handler:")
        traceback.print_exc()
        return {"error": str(exc), "trace": traceback.format_exc()}

# ──────────────────────────────────────────────────────────────────────────────
#  RUN SERVERLESS
# ──────────────────────────────────────────────────────────────────────────────
print("🚀 Starting RunPod serverless handler...")
start({"handler": handler})