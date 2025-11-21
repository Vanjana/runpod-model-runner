from steps.pipeline_step import PipelineStep  # Basis-Step-Klasse
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
import os

# Pipeline-spezifische globale Instanz
_sdxl_refiner_pipe = None

def get_sdxl_refiner_pipeline():
  global _sdxl_refiner_pipe

  if _sdxl_refiner_pipe is None:
    model_path = os.environ.get("MODEL_PATH", "/app/models/stable-diffusion-xl-refiner")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading StableDiffusionXL Refiner Pipeline from {model_path} on {device}")
    _sdxl_refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
      model_path, torch_dtype=torch_dtype, local_files_only=True
    )
    _sdxl_refiner_pipe = _sdxl_refiner_pipe.to(device)
  return _sdxl_refiner_pipe

class RefinerSDXLStep(PipelineStep):
  def run(self, input_data):
    pipe = get_sdxl_refiner_pipeline()

    # Parameter aus input_data
    init_image = input_data.get("init_image")
    if init_image is None:
      raise ValueError("init_image must be provided for SDXL Refiner step")

    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', init_image.width))
    image_height = int(input_data.get('height', init_image.height))
    inference_steps = input_data.get('inference_steps', 20)
    ai_creativity = input_data.get('ai_creativity', 7.5)
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))

    # Clamp / Validierung
    image_width = max(256, min(image_width, 2048))
    image_height = max(256, min(image_height, 2048))
    ai_creativity = max(1.0, min(ai_creativity, 20.0))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]

    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))

    # Generator für reproduzierbare Ergebnisse
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Image verfeinern
    image = pipe(
      prompt=final_positive,
      negative_prompt=final_negative,
      image=init_image,
      width=image_width,
      height=image_height,
      num_inference_steps=inference_steps,
      guidance_scale=ai_creativity,
      generator=generator
    ).images[0]

    return {
      "image": image,
      "prompt_positive": positive_prompt,
      "prompt_negative": negative_prompt,
      "prompt_positive_full": final_positive,
      "prompt_negative_full": final_negative,
      "preset_positive": positive_magic,
      "preset_negative": negative_magic,
      "width": image_width,
      "height": image_height,
      "inference_steps": inference_steps,
      "ai_creativity": ai_creativity,
      "seed": seed,
      "status": "progress"
    }
