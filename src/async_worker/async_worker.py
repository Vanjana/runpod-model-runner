from async_worker.s3_pipeline_client import S3PipelineClient
from pipelines.pipeline_factory import PipelineFactory

class AsyncWorker:
  def __init__(self, user_id, job_id, input_data):
    self.user_id = user_id
    self.job_id = job_id
    self.input_data = input_data
    self.client = S3PipelineClient()

  def run(self):
    # Status starten
    self.client.write_json(self.user_id, self.job_id, "status.json", {"status": "RUNNING"})

    try:
      # Load reference images from S3 directly into memory
      reference_images = self._load_reference_images()
      if reference_images:
        self.input_data['reference_images'] = reference_images

      # Pipeline anhand des Namens auswählen
      pipeline_name = self.input_data.get( 'pipeline_name', 'qwen' )
      pipeline = self.get_pipeline_by_name( pipeline_name )

      # Pipeline ausführen
      result = pipeline.run( self.input_data )

      # Status + Result speichern
      self.client.write_json( self.user_id, self.job_id, "status.json", {**result, "status": "FINISHED"} )

    except Exception as e:
      self.client.write_json( self.user_id, self.job_id, "status.json", {"status": "FAILED", "error": str(e)} )

  def _load_reference_images(self) -> dict:
    """
    Loads all reference images from S3 directly into memory as PIL Images
    Returns dict with keys: 'characters', 'setting', 'objects' 
    Each entry contains: {'image': PIL.Image, 'strength': float}
    """
    result = {}
    user_id = self.input_data.get('user_id', 'default')

    # Load character references
    character_refs = self.input_data.get('character_references', [])
    if character_refs:
      result['characters'] = []
      for i, ref in enumerate(character_refs):
        image_id = ref.get('imageId') if isinstance(ref, dict) else ref
        strength = ref.get('strength', 0.7) if isinstance(ref, dict) else 0.7
        image = self.client.load_image_to_memory(user_id, image_id)
        if image:
          result['characters'].append({'image': image, 'strength': strength})
          print(f"✅ Loaded character reference {i+1}/{len(character_refs)}: {image_id} (strength: {strength})")
        else:
          print(f"⚠️  Skipping missing character reference: {image_id}")

    # Load setting reference
    setting_ref = self.input_data.get('setting_reference')
    if setting_ref:
      image_id = setting_ref.get('imageId') if isinstance(setting_ref, dict) else setting_ref
      strength = setting_ref.get('strength', 0.5) if isinstance(setting_ref, dict) else 0.5
      image = self.client.load_image_to_memory(user_id, image_id)
      if image:
        result['setting'] = {'image': image, 'strength': strength}
        print(f"✅ Loaded setting reference: {image_id} (strength: {strength})")

    # Load object references
    object_refs = self.input_data.get('object_references', [])
    if object_refs:
      result['objects'] = []
      for i, ref in enumerate(object_refs):
        image_id = ref.get('imageId') if isinstance(ref, dict) else ref
        strength = ref.get('strength', 0.6) if isinstance(ref, dict) else 0.6
        image = self.client.load_image_to_memory(user_id, image_id)
        if image:
          result['objects'].append({'image': image, 'strength': strength})
          print(f"✅ Loaded object reference {i+1}/{len(object_refs)}: {image_id} (strength: {strength})")

    return result if result else None

  def get_pipeline_by_name(self, name: str):
    return PipelineFactory.get_pipeline_by_name(name)

