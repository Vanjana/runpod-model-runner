from pipeline_step import PipelineStep
from io import BytesIO
import base64

class FinalizeStep(PipelineStep):
  def run(self, input_data: dict) -> dict:
    image = input_data.get("image")

    if image is None:
      return {**input_data, "status": "error", "error": "No image to finalize"}

    # PIL-Image → Base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {**input_data, "image": image_b64, "status": "success"}
