from steps.pipeline_step import PipelineStep
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
import torch

_qwen_d11_pipe = None

def get_qwen_d11_pipeline(enable_cpu_offload=False, cpu_offload_blocks=None, pin_memory=True):
  """
  Lädt das Qwen-Image DFloat11 Modell (komprimiert).
  
  Args:
    enable_cpu_offload: CPU Offloading aktivieren (für Single-GPU <32GB)
    cpu_offload_blocks: Anzahl der Transformer-Blöcke zum Offloaden (None = alle)
    pin_memory: Memory Pinning für schnellere CPU<->GPU Transfers
  
  VRAM Requirements (Single GPU):
    - Ohne CPU Offload: ~28-32 GB
    - Mit CPU Offload: ~16 GB
  
  Multi-GPU (2x A40):
    - Automatische Verteilung über device_map="auto"
    - CPU Offload wird deaktiviert für bessere Performance
  """
  global _qwen_d11_pipe
  
  if _qwen_d11_pipe is None:
    if not torch.cuda.is_available():
      raise RuntimeError("Qwen-Image requires CUDA for bfloat16.")
    
    model_name = "Qwen/Qwen-Image"
    dfloat_model = "DFloat11/Qwen-Image-DF11"
    
    # Multi-GPU Detection
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
      print(f"Loading Qwen-Image DFloat11 with Multi-GPU support ({num_gpus} GPUs detected)")
      enable_cpu_offload = False  # Deaktiviere CPU Offload bei Multi-GPU
    else:
      print(f"Loading Qwen-Image with DFloat11 compression (Single GPU)")
      print(f"CPU Offload: {enable_cpu_offload}, Blocks: {cpu_offload_blocks}, Pin Memory: {pin_memory}")
    
    # Transformer ohne Gewichte initialisieren
    with no_init_weights():
      transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
          model_name, 
          subfolder="transformer",
        ),
      ).to(torch.bfloat16)
    
    # DFloat11 Modell laden (komprimiert)
    DFloat11Model.from_pretrained(
      dfloat_model,
      device="cpu",
      cpu_offload=enable_cpu_offload,
      cpu_offload_blocks=cpu_offload_blocks,
      pin_memory=pin_memory,
      bfloat16_model=transformer,
    )
    
    # Pipeline mit dem komprimierten Transformer erstellen
    _qwen_d11_pipe = DiffusionPipeline.from_pretrained(
      model_name,
      transformer=transformer,
      torch_dtype=torch.bfloat16,
    )
    
    # Bei Multi-GPU: Komponenten manuell verteilen
    if use_multi_gpu:
      # Text encoder auf GPU 0
      if hasattr(_qwen_d11_pipe, 'text_encoder') and _qwen_d11_pipe.text_encoder is not None:
        _qwen_d11_pipe.text_encoder = _qwen_d11_pipe.text_encoder.to("cuda:0")
      # Transformer (DFloat11) auf GPU 1
      _qwen_d11_pipe.transformer = _qwen_d11_pipe.transformer.to("cuda:1")
      # VAE auf GPU 1
      if hasattr(_qwen_d11_pipe, 'vae') and _qwen_d11_pipe.vae is not None:
        _qwen_d11_pipe.vae = _qwen_d11_pipe.vae.to("cuda:1")
      vram_mode = f"distributed across {num_gpus} GPUs"
    else:
      # Single GPU: enable_model_cpu_offload nutzen
      _qwen_d11_pipe.enable_model_cpu_offload()
      vram_mode = "~16 GB VRAM" if enable_cpu_offload else "~28 GB VRAM"
    
    print(f"Qwen-Image DFloat11 Pipeline loaded successfully ({vram_mode})")
  
  return _qwen_d11_pipe

class GenerateQwenD11Step(PipelineStep):
  def run(self, input_data):
    # CPU Offload Konfiguration aus input_data (oder Default)
    # Bei Multi-GPU wird CPU Offload automatisch deaktiviert
    enable_cpu_offload = input_data.get('cpu_offload', False)  # Default: deaktiviert
    cpu_offload_blocks = input_data.get('cpu_offload_blocks', None)  # None = alle
    pin_memory = input_data.get('pin_memory', True)
    
    pipe = get_qwen_d11_pipeline(
      enable_cpu_offload=enable_cpu_offload,
      cpu_offload_blocks=cpu_offload_blocks,
      pin_memory=pin_memory
    )
    
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
    
    # Generator auf dem Device der Pipeline (wichtig bei Multi-GPU!)
    # Bei Multi-GPU ist Transformer auf cuda:1
    try:
      if hasattr(pipe, 'transformer') and pipe.transformer is not None:
        device = str(pipe.transformer.device)
      else:
        device = str(next(pipe.parameters()).device)
    except:
      device = "cuda"
    generator = torch.Generator(device).manual_seed(seed)
    
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
      "cpu_offload": enable_cpu_offload,
      "language": language,
      "peak_vram_gb": max_memory / (1024**3),
      "model_type": "DFloat11"
    }
