from pipeline_step import PipelineStep
from PIL import Image
import io
import base64
import os
import threading

# Third-party
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# -----------------------------
# Thread-safe Model Cache
# -----------------------------
class _ModelCache:
  def __init__(self):
    self._lock = threading.RLock()
    self._upsamplers = {}

  def get_upsampler(self, model_name, netscale, rrdb_kwargs, model_path, gpu_id, **realesrgan_kwargs):
    key = f"{model_name}:{netscale}:{model_path}:{gpu_id}"
    with self._lock:
      if key in self._upsamplers:
        return self._upsamplers[key]

      rrdb = RRDBNet(**rrdb_kwargs)
      if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

      upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=rrdb,
        tile=realesrgan_kwargs.get('tile', 0),
        tile_pad=realesrgan_kwargs.get('tile_pad', 10),
        pre_pad=realesrgan_kwargs.get('pre_pad', 0),
        half=realesrgan_kwargs.get('half', False),
        gpu_id=gpu_id
      )

      self._upsamplers[key] = upsampler
      return upsampler

# globaler Cache
_model_cache = _ModelCache()