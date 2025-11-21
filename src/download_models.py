#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub import login
import os
import sys

# -----------------------------------------------------------------------
# Zielverzeichnis
# -----------------------------------------------------------------------
DEFAULT_MODEL_DIR = Path("/app/models")
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# MODEL REGISTRY
# gui_name → info: { hf_id, type, description }
# -----------------------------------------------------------------------
MODEL_REGISTRY = {
    # SDXL
    "sdxl-base": {"hf_id": "stabilityai/stable-diffusion-xl-base-1.0", "type": "sdxl-base", "description": "Stable Diffusion XL Base Model 1.0"},
    "sdxl-refiner": {"hf_id": "stabilityai/stable-diffusion-xl-refiner-1.0", "type": "sdxl-refiner", "description": "Stable Diffusion XL Refiner Model 1.0"},
    # Qwen
    "qwen-image": {"hf_id": "microsoft/Qwen-Image-Base", "type": "qwen-image", "description": "Qwen Image Base Model"},
    "qwen-image-edit": {"hf_id": "microsoft/Qwen-Image-Edit", "type": "qwen-image-edit", "description": "Qwen Image Editing Model"},
    # RealESRGAN
    "realesrgan-x4plus": {"hf_id": "xinntao/RealESRGAN_x4plus", "type": "realesrgan", "description": "RealESRGAN Upscaler x4"},
    "realesrgan-x2": {"hf_id": "xinntao/RealESRGAN_x2plus", "type": "realesrgan", "description": "RealESRGAN Upscaler x2"},
    "realesrgan-anime": {"hf_id": "xinntao/RealESRGAN_animevideov3", "type": "realesrgan", "description": "RealESRGAN Anime Upscaler"},
    "realesrgan-face": {"hf_id": "xinntao/RealESRGANv2-Face", "type": "realesrgan", "description": "RealESRGAN Face Enhancer"},
}

# -----------------------------------------------------------------------
# LOADER-FUNKTIONEN
# -----------------------------------------------------------------------
def load_sdxl_base(model_id, cache_dir):
    return StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)

def load_sdxl_refiner(model_id, cache_dir):
    return StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)

def load_qwen_image(model_id, cache_dir):
    from transformers import AutoModel, AutoProcessor
    AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

def load_qwen_image_edit(model_id, cache_dir):
    from transformers import AutoModel, AutoProcessor
    AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

def load_realesrgan(model_id, cache_dir):
    from realesrgan import RealESRGAN
    RealESRGAN.from_pretrained(model_id, cache_dir=cache_dir)

# Zuordnung Loader
LOADERS = {
    "sdxl-base": load_sdxl_base,
    "sdxl-refiner": load_sdxl_refiner,
    "qwen-image": load_qwen_image,
    "qwen-image-edit": load_qwen_image_edit,
    "realesrgan": load_realesrgan,
}

# -----------------------------------------------------------------------
# Download-Funktion
# -----------------------------------------------------------------------
def download_model(model_key, target_dir):
    if model_key not in MODEL_REGISTRY:
        print(f"❌ Modell '{model_key}' nicht bekannt.")
        print("   Nutze --list für eine Übersicht.")
        sys.exit(1)

    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["hf_id"]
    model_type = model_info["type"]

    print(f"⬇️ Lade Modell '{model_key}' ({model_id}) nach '{target_dir}'...")

    if model_type not in LOADERS:
        print(f"❌ Kein Loader für Modelltyp '{model_type}' definiert.")
        sys.exit(1)

    loader_fn = LOADERS[model_type]
    loader_fn(model_id, cache_dir=target_dir)

    print(f"✔️ '{model_key}' erfolgreich heruntergeladen!\n")

# -----------------------------------------------------------------------
# CLI Parser
# -----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Download-Tool für KI-Modelle (SDXL, Qwen, RealESRGAN) in /app/models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, help="GUI-Name des Modells, das heruntergeladen werden soll")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Zielverzeichnis für die Modelle")
    parser.add_argument("--token", type=str, default=None, help="Optional: HuggingFace Token (oder via HF_TOKEN Umgebungsvariable)")
    parser.add_argument("--list", action="store_true", help="Zeige alle verfügbaren Modelle")

    return parser

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    target_dir = Path(args.dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("📦 Verfügbare Modelle:")
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:<20} → {info['hf_id']} - {info['description']}")
        return

    if not args.model:
        print("❌ Bitte ein Modell mit --model angeben oder --list verwenden.")
        sys.exit(1)

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        print("🔐 Login in HuggingFace...")
        login(token=token)
    else:
        print("ℹ️ Kein Token angegeben (öffentliche Modelle funktionieren trotzdem).")

    download_model(args.model, target_dir)

if __name__ == "__main__":
    main()
