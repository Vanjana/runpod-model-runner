from steps.pipeline_step import PipelineStep
from diffusers import StableDiffusionPipeline
import torch

_sd_pipelines = {}

def get_sd_pipeline(model_name="runwayml/stable-diffusion-v1-5"):
  if model_name not in _sd_pipelines:
    if not torch.cuda.is_available():
      raise RuntimeError("Stable Diffusion requires CUDA for FP16.")
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
      print(f"Loading {model_name} with Multi-GPU support ({num_gpus} GPUs)")
      pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map="balanced"  # Balanced distribution across GPUs
      )
    else:
      print(f"Loading {model_name} on single GPU")
      pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True
      )
      pipe = pipe.to("cuda")
    
    _sd_pipelines[model_name] = pipe
  return _sd_pipelines[model_name]

class GenerateSDMultiStep(PipelineStep):
  def run(self, input_data):
    model_name = input_data.get("model_name", "runwayml/stable-diffusion-v1-5")
    pipe = get_sd_pipeline(model_name)

    positive_prompt = input_data.get("prompt_positive", "")
    negative_prompt = input_data.get("prompt_negative", "")
    width = max(256, min(int(input_data.get("width", 512)), 1024))
    height = max(256, min(int(input_data.get("height", 512)), 1024))
    steps = input_data.get("inference_steps", 20)
    guidance = input_data.get("ai_creativity", 7.5)
    seed = int(input_data.get("seed", torch.randint(0, 2**32 - 1, (1,)).item()))

    # Generator - CPU verwenden für Multi-GPU Kompatibilität
    generator = torch.Generator().manual_seed(seed)

    # Inference (Pipeline ist bereits in fp16, kein autocast nötig)
    with torch.inference_mode():
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
      "model_name": model_name,
      "prompt_positive": positive_prompt,
      "prompt_negative": negative_prompt,
      "width": width,
      "height": height,
      "inference_steps": steps,
      "ai_creativity": guidance,
      "seed": seed
    }
