from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_zimage import GenerateZImageStep

pipeline = Pipeline(steps=[
    GenerateZImageStep(),
    FinalizeStep()
])
