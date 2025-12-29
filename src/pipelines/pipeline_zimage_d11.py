from pipelines.pipeline import Pipeline
from steps.generate_zimage_d11 import GenerateZImageD11Step
from steps.finalize import FinalizeStep

pipeline = Pipeline([
  GenerateZImageD11Step(),
  FinalizeStep()
])
