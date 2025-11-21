import threading
import uuid
from flask import Flask, request, jsonify
from async_worker.s3_pipeline_client import S3PipelineClient
from async_worker.async_worker import AsyncWorker

class JsonServer:
  def __init__(self, host="0.0.0.0", port=8000):
    self.app = Flask(__name__)
    self.client = S3PipelineClient()
    self.host = host
    self.port = port

    # Mutex/Semaphore für "nur 1 Job gleichzeitig"
    self.job_lock = threading.Lock()

    # Endpoints registrieren
    self._register_routes()

  def _register_routes(self):
    @self.app.route("/run", methods=["POST"])
    def run():
      data = request.get_json()

      if not data or "user_id" not in data or "input" not in data:
        return jsonify({"error": "Missing user_id or input"}), 400

      user_id = data["user_id"]
      input_data = data["input"]
      job_id = str(uuid.uuid4())

      # Initialer Status in S3
      self.client.write_json(user_id, job_id, "request.json", {"input": input_data})
      self.client.write_json(user_id, job_id, "status.json", {"status": "PENDING"})

      # Async Worker starten
      threading.Thread(
        target=self._run_job, args=(user_id, job_id, input_data), daemon=True
      ).start()

      return jsonify({"job_id": job_id, "status": "PENDING"})

    @self.app.route("/status/<user_id>/<job_id>", methods=["GET"])
    def get_status(user_id, job_id):
      status = self.client.read_json(user_id, job_id, "status.json")
      if status is None:
        return jsonify({"error": "Job not found"}), 404
      return jsonify(status)

    @self.app.route("/result/<user_id>/<job_id>", methods=["GET"])
    def get_result(user_id, job_id):
      result = self.client.read_json(user_id, job_id, "result.json")
      if result is None:
        return jsonify({"status": "PENDING"}), 202
      return jsonify(result)

  def _run_job(self, user_id, job_id, input_data):
    """Hilfsfunktion, die Worker im Thread startet."""
    # Hier verhindern wir parallele Ausführung
    with self.job_lock:
      worker = AsyncWorker(user_id, job_id, input_data)
      worker.run()

  def start(self):
    """Starte den Flask-Server."""
    self.app.run(host=self.host, port=self.port)
