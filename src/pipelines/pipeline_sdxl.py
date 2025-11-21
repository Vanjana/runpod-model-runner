from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_sdxl.py import GenerateSDXLStep

pipeline = Pipeline(steps=[
    GenerateSDXLStep(),
    FinalizeStep()
])
