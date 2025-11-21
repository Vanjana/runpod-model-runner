class PipelineStep:
  def __init__(self, steps):
    self.steps = steps

  def run(self, input_data):
    data = input_data

    for step in self.steps:
      print(f"Running step: {step.__class__.__name__}")
      data = step.run(data)

      if data.get("status") == "error":
        return data

    return data
