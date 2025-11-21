from steps.upscale_realESRGAN import UpscaleRealESRGANStep

class UpscaleRealESRGAN_AnimeStep(UpscaleRealESRGANStep):
  model_name = "RealESRGAN_x4plus_anime_6B"

  def get_rrdb_args(self):
    return dict(
      num_in_ch=3,
      num_out_ch=3,
      num_feat=64,
      num_block=6,
      num_grow_ch=32
    )

  def get_scale(self):
    return 4

  def get_model_path(self):
    return "models/RealESRGAN_x4plus_anime_6B.pth"