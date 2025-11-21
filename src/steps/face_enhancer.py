from steps.pipeline_step import PipelineStep

_face_model_cache = {}

class FaceEnhancerBaseStep(PipelineStep):

  model_name = None  # Muss gesetzt werden
  model_path = None  # Muss gesetzt werden

  def __init__(self):
    super().__init__(self.model_name)

  def load_model(self):
    """Subklassen müssen GFPGAN/CodeFormer/etc. initialisieren."""
    raise NotImplementedError

  def _get_model(self):
    if self.model_name in _face_model_cache:
      return _face_model_cache[self.model_name]

    model = self.load_model()
    _face_model_cache[self.model_name] = model
    return model

  def run(self, data: dict) -> dict:
    img = data["image"]

    model = self._get_model()

    enhanced = self.enhance(model, img)

    return {**data, "image": enhanced}

  def enhance(self, model, img):
    """Subklassen implementieren die eigentliche Verarbeitung."""
    raise NotImplementedError
