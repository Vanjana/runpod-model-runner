from async_worker.s3_pipeline_client import S3PipelineClient


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
      # Pipeline anhand des Namens auswählen
      pipeline_name = self.input_data.get( 'pipeline_name', 'qwen' )
      pipeline = self.get_pipeline_by_name( pipeline_name )

      # Pipeline ausführen
      result = pipeline.run( self.input_data )

      # Status + Result speichern
      self.client.write_json( self.user_id, self.job_id, "status.json", {"status": "FINISHED", **result} )

    except Exception as e:
      self.client.write_json( self.user_id, self.job_id, "status.json", {"status": "FAILED", "error": str(e)} )

  def get_pipeline_by_name(self, name: str):
    from pipelines import pipeline_qwen
    return pipeline_qwen
