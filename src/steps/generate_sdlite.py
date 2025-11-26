from steps.pipeline_step import PipelineStep
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast

_sd_lite_pipe = None

def get_sd_lite_pipeline():
  global _sd_lite_pipe
  if _sd_lite_pipe is None:
    if not torch.cuda.is_available():
      raise RuntimeError("Stable Diffusion Lite requires CUDA for FP16.")
    device = "cuda"
    print("Loading Stable Diffusion Lite on", device)
    _sd_lite_pipe = StableDiffusionPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      torch_dtype=torch.float16,
      use_safetensors=True
    )
    _sd_lite_pipe = _sd_lite_pipe.to(device)
  return _sd_lite_pipe

class GenerateSDLiteStep(PipelineStep):
  def run(self, input_data):
    pipe = get_sd_lite_pipeline()

    positive_prompt = input_data.get("prompt_positive", "")
    negative_prompt = input_data.get("prompt_negative", "")
    width = max(256, min(int(input_data.get("width", 512)), 768))
    height = max(256, min(int(input_data.get("height", 512)), 768))
    steps = input_data.get("inference_steps", 20)
    guidance = input_data.get("ai_creativity", 7.5)
    seed = int(input_data.get("seed", torch.randint(0, 2**32 - 1, (1,)).item()))

    generator = torch.Generator("cuda").manual_seed(seed)
    device = "cuda"

    with torch.inference_mode(), autocast(device_type=device, dtype=torch.float16):
      image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
      ).images[0]

    torch.cuda.empty_cache()

    return {
      "image": image,
      "prompt_positive": positive_prompt,
      "prompt_negative": negative_prompt,
      "width": width,
      "height": height,
      "inference_steps": steps,
      "ai_creativity": guidance,
      "seed": seed
    }
