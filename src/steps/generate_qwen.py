from pipeline_step import PipelineStep  # Basis-Step-Klasse
from diffusers import DiffusionPipeline
import torch
import os

# Pipeline-spezifische globale Instanz
_qwen_image_pipe = None

def get_pipeline():
  global _qwen_image_pipe

  if _qwen_image_pipe is None:
    model_path = os.environ.get("MODEL_PATH", "/app/models/qwen-image")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen Image Pipeline from {model_path} on {device}")
    _qwen_image_pipe = DiffusionPipeline.from_pretrained(
      model_path, dtype=torch_dtype, local_files_only=True
    )
    _qwen_image_pipe = _qwen_image_pipe.to(device)
  return _qwen_image_pipe

class GenerateQwenStep(PipelineStep):
  def run(self, input_data):
    pipe = get_pipeline()
    
    # Parameter aus input_data
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', 1328))
    image_height = int(input_data.get('height', 1328))
    inference_steps = input_data.get('inference_steps', 20)
    ai_creativity = input_data.get('ai_creativity', 4.0)
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))

    # Clamp / Validierung
    image_width = max(256, min(image_width, 1584))
    image_height = max(256, min(image_height, 1584))
    ai_creativity = max(1.0, min(ai_creativity, 6.0))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]

    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))

    # Generator für reproduzierbare Ergebnisse
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Image generieren
    image = pipe(
      prompt=final_positive,
      negative_prompt=final_negative,
      width=image_width,
      height=image_height,
      num_inference_steps=inference_steps,
      true_cfg_scale=ai_creativity,
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
