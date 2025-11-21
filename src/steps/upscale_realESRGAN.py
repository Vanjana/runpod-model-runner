from basicsr.archs.rrdbnet_arch import RRDBNet
from pipeline_step import PipelineStep
from realesrgan import RealESRGANer

_model_cache = {}

class UpscaleRealESRGANStep(PipelineStep):

    model_name = None  # Muss von Subklassen überschrieben werden

    def __init__(self):
        super().__init__(self.model_name)

    # Muss von Subklassen implementiert werden
    def get_rrdb_args(self) -> dict:
        raise NotImplementedError

    # Muss gesetzt werden: Upscaling factor (2 oder 4)
    def get_scale(self) -> int:
        raise NotImplementedError

    # Optional falls Modelle lokale Pfade nutzen:
    def get_model_path(self) -> str:
        raise NotImplementedError

    def _load_model(self):
        if self.model_name in _model_cache:
            return _model_cache[self.model_name]

        # Parameter vom Subclass
        rrdb_args = self.get_rrdb_args()
        scale = self.get_scale()
        model_path = self.get_model_path()

        model = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=RRDBNet(**rrdb_args),
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,  # für CPU/Kompatibilität
        )

        _model_cache[self.model_name] = model
        return model

    def run(self, data: dict) -> dict:
        image = data["image"]

        upscaler = self._load_model()
        output, _ = upscaler.enhance(image)

        return {**data, "image": output}
