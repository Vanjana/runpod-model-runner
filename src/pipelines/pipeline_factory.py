from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.face_enhancer_GFPGAN import GFPGANFaceEnhancer
from steps.generate_qwen import GenerateQwenStep
from steps.upscale_realESRGAN_anime import UpscaleRealESRGAN_AnimeStep
from steps.upscale_realESRGAN_x2 import UpscaleRealESRGAN_X2Step
from steps.upscale_realESRGAN_x4 import UpscaleRealESRGAN_X4Step


class PipelineFactory:
  REGISTRY = {
    "GenerateQwenStep": GenerateQwenStep,
    "GFPGANFaceEnhancer": GFPGANFaceEnhancer,
    "UpscaleRealESRGAN_X4Step": UpscaleRealESRGAN_X4Step,
    "UpscaleRealESRGAN_X2Step": UpscaleRealESRGAN_X2Step,
    "UpscaleRealESRGAN_AnimeStep": UpscaleRealESRGAN_AnimeStep,
  }

  @classmethod
  def from_json(cls, pipeline_json):
    steps = []
    for step_def in pipeline_json["steps"]:
      step_type = step_def["type"]
      step_cls = cls.REGISTRY[step_type]
      steps.append(step_cls())
    steps.append(FinalizeStep())
    return Pipeline(steps)
