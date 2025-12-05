from steps.pipeline_step import PipelineStep
from diffusers import DiffusionPipeline
import torch
from torch import autocast

_qwen_image_pipe = None

def get_pipeline():
  global _qwen_image_pipe
  if _qwen_image_pipe is None:
    model_name = "Qwen/Qwen-Image"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16  # statt bfloat16

    print(f"Loading Qwen-Image Pipeline on {device} with {torch_dtype}")

    _qwen_image_pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    _qwen_image_pipe = _qwen_image_pipe.to(device)

  return _qwen_image_pipe

class GenerateQwenStep(PipelineStep):
  def run(self, input_data):
    pipe = get_pipeline()

    # Parameter
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', 512))
    image_height = int(input_data.get('height', 512))
    inference_steps = input_data.get('inference_steps', 20)
    ai_creativity = input_data.get('ai_creativity', 4.0)
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))

    # Clamp / Validierung
    image_width = max(256, min(image_width, 1024))   # sicherer für VRAM
    image_height = max(256, min(image_height, 1024))
    ai_creativity = max(1.0, min(ai_creativity, 6.0))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]

    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Image generieren in inference mode + autocast für float16
    with torch.inference_mode(), autocast(device_type=device, dtype=torch.float16):
      image = pipe(
        prompt=final_positive,
        negative_prompt=final_negative,
        width=image_width,
        height=image_height,
        num_inference_steps=inference_steps,
        true_cfg_scale=ai_creativity,
        generator=generator
      ).images[0]

    # GPU Cache leeren nach jedem Run (optional)
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
