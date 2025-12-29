from steps.pipeline_step import PipelineStep  # Basis-Step-Klasse
from diffusers import AutoPipelineForText2Image
import torch

# Pipeline-spezifische globale Instanz
_sdxl_turbo_pipe = None

def get_sdxl_turbo_pipeline():
  global _sdxl_turbo_pipe

  if _sdxl_turbo_pipe is None:
    model_name = "stabilityai/sdxl-turbo"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
      print(f"Loading SDXL Turbo Pipeline with Multi-GPU support ({num_gpus} GPUs)")
      _sdxl_turbo_pipe = AutoPipelineForText2Image.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        variant="fp16",
        device_map="balanced"  # Balanced distribution across GPUs
      )
    else:
      print(f"Loading SDXL Turbo Pipeline on single GPU")
      _sdxl_turbo_pipe = AutoPipelineForText2Image.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        variant="fp16"
      )
      _sdxl_turbo_pipe = _sdxl_turbo_pipe.to("cuda")

  return _sdxl_turbo_pipe


class GenerateSDXLTurboStep(PipelineStep):
  def run(self, input_data):
    pipe = get_sdxl_turbo_pipeline()

    # Parameter aus input_data
    positive_magic = input_data.get('preset_positive', [])
    negative_magic = input_data.get('preset_negative', [])
    positive_prompt = input_data.get('prompt_positive', '')
    negative_prompt = input_data.get('prompt_negative', '')
    image_width = int(input_data.get('width', 1024))
    image_height = int(input_data.get('height', 1024))
    inference_steps = input_data.get('inference_steps', 4)  # Turbo meist 1–4 Steps
    seed = int(input_data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item()))

    # Validierung
    image_width = max(256, min(image_width, 2048))
    image_height = max(256, min(image_height, 2048))
    if isinstance(positive_magic, str):
      positive_magic = [positive_magic]
    if isinstance(negative_magic, str):
      negative_magic = [negative_magic]

    final_positive = " ".join(filter(None, [positive_prompt] + positive_magic))

    # Turbo ignoriert negative_prompt in vielen Fällen → aber wir geben es trotzdem durch
    final_negative = " ".join(filter(None, [negative_prompt] + negative_magic))

    # Generator auf dem Device der Pipeline
    device = next(pipe.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    # Turbo unterstützt *keine* guidance_scale -> wird einfach ignoriert
    image = pipe(
      prompt=final_positive,
      negative_prompt=final_negative,
      width=image_width,
      height=image_height,
      num_inference_steps=inference_steps,
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
      "seed": seed
    }
