from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_sdxl_turbo import GenerateSDXLTurboStep

pipeline = Pipeline(steps=[
  GenerateSDXLTurboStep(),
  FinalizeStep()
])
