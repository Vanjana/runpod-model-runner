from steps.pipeline_step import PipelineStep
from diffusers import DiffusionPipeline
import torch

_sdxl_pipe = None

def get_sdxl_pipeline():
  global _sdxl_pipe

  if _sdxl_pipe is None:
    if not torch.cuda.is_available():
      raise RuntimeError("SDXL requires CUDA for fp16. CPU does not support Half precision.")

    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
      print(f"Loading StableDiffusionXL Pipeline with Multi-GPU support ({num_gpus} GPUs)")
      _sdxl_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        device_map="balanced"  # Balanced distribution across GPUs
      )
    else:
      print(f"Loading StableDiffusionXL Pipeline on single GPU")
      _sdxl_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
      )
      _sdxl_pipe = _sdxl_pipe.to("cuda")

  return _sdxl_pipe

class GenerateSDXLStep(PipelineStep):
  def run(self, input_data):
    pipe = get_sdxl_pipeline()

    # Parameter
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', 1024))
    image_height = int(input_data.get('height', 1024))
    inference_steps = input_data.get('inference_steps', 20)
    ai_creativity = input_data.get('ai_creativity', 7.5)
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))

    # Clamp / Validierung
    image_width = max(256, min(image_width, 1536))   # sicherer für VRAM
    image_height = max(256, min(image_height, 1536))
    ai_creativity = max(1.0, min(ai_creativity, 20.0))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]

    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))

    # Generator auf dem Device der Pipeline
    device = next(pipe.parameters()).device
    generator = torch.Generator(device).manual_seed(seed)

    # Inference Mode (ohne autocast - Pipeline ist bereits in fp16)
    with torch.inference_mode():
      image = pipe(
        prompt=final_positive,
        negative_prompt=final_negative,
        width=image_width,
        height=image_height,
        num_inference_steps=inference_steps,
        guidance_scale=ai_creativity,
        generator=generator
      ).images[0]

    # Optional: GPU Cache leeren
    torch.cuda.empty_cache()

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
      "seed": seed
    }
