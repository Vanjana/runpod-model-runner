from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_sd15 import GenerateSDMultiStep

pipeline = Pipeline(steps=[
    GenerateSDMultiStep(),
    FinalizeStep()
])
