from gfpgan import GFPGANer
from steps.face_enhancer import FaceEnhancerBaseStep

class GFPGANFaceEnhancer(FaceEnhancerBaseStep):
  model_name = "GFPGANv1.4"
  model_path = "models/GFPGANv1.4.pth"

  def load_model(self):
    # Falls du RealESRGAN einbinden willst:
    # → Kann optional wieder aus deinem UpscalerCache kommen
    upsampler = None

    return GFPGANer(
      model_path=self.model_path,
      upscale=1,
      arch="clean",
      channel_multiplier=2,
      bg_upsampler=upsampler,
    )

  def enhance(self, model: GFPGANer, img):
    cropped_faces, restored_faces, restored_img = model.enhance(
      img,
      has_aligned=False,
      only_center_face=False,
      paste_back=True
    )
    return restored_img
