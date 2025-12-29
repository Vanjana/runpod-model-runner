from steps.pipeline_step import PipelineStep
from diffusers import DiffusionPipeline
import torch

_qwen_pipe = None

def get_qwen_pipeline():
  """
  Lädt das normale Qwen-Image Modell (ohne DFloat11 Kompression).
  
  VRAM Requirements:
    - Single GPU: ~41 GB
    - Multi-GPU (2x A40): Automatische Verteilung
  """
  global _qwen_pipe
  
  if _qwen_pipe is None:
    if not torch.cuda.is_available():
      raise RuntimeError("Qwen-Image requires CUDA for bfloat16.")
    
    model_name = "Qwen/Qwen-Image"
    
    # Multi-GPU Detection
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
      print(f"Loading Qwen-Image (full model) with Multi-GPU support ({num_gpus} GPUs detected)")
      # Bei Multi-GPU: balanced device_map für automatische Verteilung
      _qwen_pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
      )
      vram_mode = f"distributed across {num_gpus} GPUs"
    else:
      print(f"Loading Qwen-Image (full model) on single GPU")
      _qwen_pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
      )
      _qwen_pipe = _qwen_pipe.to("cuda")
      vram_mode = "~41 GB VRAM"
    
    print(f"Qwen-Image Pipeline loaded successfully ({vram_mode})")
  
  return _qwen_pipe

class GenerateQwenStep(PipelineStep):
  def run(self, input_data):
    pipe = get_qwen_pipeline()
    
    # Parameter
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    
    # Aspect Ratio statt width/height
    aspect_ratio = input_data.get('aspect_ratio', '16:9')
    aspect_ratios = {
      "1:1": (1328, 1328),
      "16:9": (1664, 928),
      "9:16": (928, 1664),
      "4:3": (1472, 1140),
      "3:4": (1140, 1472),
    }
    image_width, image_height = aspect_ratios.get(aspect_ratio, (1664, 928))
    
    # Andere Parameter
    inference_steps = input_data.get('inference_steps', 50)  # Qwen-Image braucht mehr Steps
    ai_creativity = input_data.get('ai_creativity', 4.0)  # true_cfg_scale
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))
    language = input_data.get('language', 'en')  # 'en' oder 'zh'
    
    # Validierung
    ai_creativity = max(1.0, min(ai_creativity, 10.0))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]
    
    # Magic Prompts (Standard für bessere Qualität)
    default_magic = {
      "en": "Ultra HD, 4K, cinematic composition.",
      "zh": "超清，4K，电影级构图"
    }
    
    # Wenn keine positive_magic angegeben, Standard verwenden
    if not positive_magic:
      positive_magic = [default_magic.get(language, default_magic["en"])]
    
    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))
    
    # Generator - bei Multi-GPU einfach "cuda" verwenden
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Inference
    print(f"Generating {aspect_ratio} image with {inference_steps} steps...")
    with torch.inference_mode():
      image = pipe(
        prompt=final_positive,
        negative_prompt=final_negative,
        width=image_width,
        height=image_height,
        num_inference_steps=inference_steps,
        true_cfg_scale=ai_creativity,
        generator=generator
      ).images[0]
    
    # VRAM Stats ausgeben
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Peak VRAM usage: {max_memory / (1024**3):.2f} GB")
    
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
      "aspect_ratio": aspect_ratio,
      "width": image_width,
      "height": image_height,
      "inference_steps": inference_steps,
      "ai_creativity": ai_creativity,
      "seed": seed,
      "language": language,
      "peak_vram_gb": max_memory / (1024**3),
      "model_type": "Full"
    }
