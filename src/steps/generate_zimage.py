from steps.pipeline_step import PipelineStep
from diffusers import ZImagePipeline
import torch

_zimage_pipe = None

def get_zimage_pipeline():
  """
  Lädt das Z-Image-Turbo Pipeline.
  
  VRAM Requirements:
    - Single GPU: ~16 GB
    - Multi-GPU (2x A40): Automatische Verteilung (~8 GB per GPU)
  
  Features:
    - Photorealistic quality
    - Bilingual text rendering (English & Chinese)
    - Sub-second generation on H800
    - 8-step inference (9 num_inference_steps)
  """
  global _zimage_pipe
  
  if _zimage_pipe is None:
    if not torch.cuda.is_available():
      raise RuntimeError("Z-Image-Turbo requires CUDA for bfloat16.")
    
    model_name = "Tongyi-MAI/Z-Image-Turbo"
    
    # Multi-GPU Detection
    num_gpus = torch.cuda.device_count()
    
    print(f"Loading Z-Image-Turbo Pipeline{' with Multi-GPU support' if num_gpus > 1 else ''}")
    
    if num_gpus > 1:
      _zimage_pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for optimal performance
        device_map="balanced",  # Balanced distribution across GPUs
        low_cpu_mem_usage=False,
      )
    else:
      _zimage_pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
      )
      _zimage_pipe = _zimage_pipe.to("cuda")
    
    # Optional: Enable Flash Attention for better efficiency
    # _zimage_pipe.transformer.set_attention_backend("flash")
    
    vram_mode = f"~{16//num_gpus} GB per GPU" if num_gpus > 1 else "~16 GB VRAM"
    print(f"Z-Image-Turbo Pipeline loaded successfully ({vram_mode})")
  
  return _zimage_pipe

class GenerateZImageStep(PipelineStep):
  def run(self, input_data):
    pipe = get_zimage_pipeline()
    
    # Parameter
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', 1024))
    image_height = int(input_data.get('height', 1024))
    inference_steps = input_data.get('inference_steps', 9)  # Z-Image-Turbo default (8 DiT forwards)
    ai_creativity = input_data.get('ai_creativity', 0.0)  # Should be 0 for Turbo models
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))
    
    # Validierung
    image_width = max(256, min(image_width, 2048))
    image_height = max(256, min(image_height, 2048))
    
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]
    
    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))
    
    # Generator - CPU verwenden für Multi-GPU Kompatibilität
    generator = torch.Generator().manual_seed(seed)
    
    # Inference (Z-Image-Turbo doesn't use negative prompts with guidance_scale=0)
    print(f"Generating {image_width}x{image_height} image with {inference_steps} steps (Z-Image-Turbo)...")
    with torch.inference_mode():
      image = pipe(
        prompt=final_positive,
        width=image_width,
        height=image_height,
        num_inference_steps=inference_steps,
        guidance_scale=ai_creativity,
        generator=generator
      ).images[0]
    
    # GPU Cache leeren
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
