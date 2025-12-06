import json
import os
import boto3
from io import BytesIO
from PIL import Image

class S3PipelineClient:
  def __init__(self, 
         bucket: str = None, 
         endpoint_url: str = None,
         aws_key: str = None,
         aws_secret: str = None):
    self.bucket = bucket or os.environ.get("S3_BUCKET")

    self.s3_client = boto3.client(
      "s3",
      endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT"),
      aws_access_key_id=aws_key or os.environ.get("S3_KEY"),
      aws_secret_access_key=aws_secret or os.environ.get("S3_SECRET"),
      region_name=os.environ.get("S3_REGION", "eu-ro-1")
    )

  # --- Hilfsfunktionen ---
  def s3_path(self, user_id: str, job_id: str, filename: str) -> str:
    """Erstellt den vollständigen S3-Pfad für ein Objekt."""
    return f"{user_id}/jobs/{job_id}/{filename}"

  def write_json(self, user_id: str, job_id: str, filename: str, data: dict):
    """Speichert ein JSON-Objekt auf S3."""
    self.s3_client.put_object(
      Bucket=self.bucket,
      Key=self.s3_path(user_id, job_id, filename),
      Body=json.dumps(data).encode("utf-8")
    )

  def read_json(self, user_id: str, job_id: str, filename: str):
    """Liest ein JSON-Objekt von S3. Gibt None zurück, falls nicht gefunden."""
    try:
      obj = self.s3_client.get_object(
        Bucket=self.bucket,
        Key=self.s3_path(user_id, job_id, filename)
      )
      return json.loads(obj['Body'].read())
    except self.s3_client.exceptions.NoSuchKey:
      return None

  def load_image_to_memory(self, user_id: str, image_id: str):
    """
    Loads an image from S3 directly into memory as PIL Image
    Images are stored at: {user_id}/uploads/{image_id}
    Returns PIL Image on success, None if not found
    """
    try:
      s3_key = f"{user_id}/uploads/{image_id}"
      response = self.s3_client.get_object(
        Bucket=self.bucket,
        Key=s3_key
      )
      image_data = response['Body'].read()
      image = Image.open(BytesIO(image_data))
      return image
    except self.s3_client.exceptions.NoSuchKey:
      print(f"⚠️  Image not found in S3: {image_id}")
      return None
    except Exception as e:
      print(f"❌ Failed to load image {image_id}: {e}")
      return None

