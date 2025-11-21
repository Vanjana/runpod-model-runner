from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep

from steps.face_enhancer_GFPGAN import GFPGANFaceEnhancer
from steps.generate_qwen import GenerateQwenStep
from steps.upscale_realESRGAN_anime import UpscaleRealESRGAN_AnimeStep
from steps.upscale_realESRGAN_x2 import UpscaleRealESRGAN_X2Step
from steps.upscale_realESRGAN_x4 import UpscaleRealESRGAN_X4Step

pipeline = Pipeline(steps=[
    GenerateQwenStep(),
    GFPGANFaceEnhancer(),
    UpscaleRealESRGAN_X4Step(),
    FinalizeStep()
])
