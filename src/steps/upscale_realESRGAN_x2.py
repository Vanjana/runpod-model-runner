from steps.upscale_realESRGAN import UpscaleRealESRGANStep

class UpscaleRealESRGAN_X2Step(UpscaleRealESRGANStep):
  model_name = "RealESRGAN_x2plus"

  def get_rrdb_args(self):
    return dict(
      num_in_ch=3,
      num_out_ch=3,
      num_feat=64,
      num_block=23,
      num_grow_ch=32
    )

  def get_scale(self):
    return 2

  def get_model_path(self):
    return "models/RealESRGAN_x2plus.pth"