from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_sdlite import GenerateSDLiteStep

pipeline = Pipeline(steps=[
    GenerateSDLiteStep(),
    FinalizeStep()
])
