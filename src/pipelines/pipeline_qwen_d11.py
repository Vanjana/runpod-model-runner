from pipelines.pipeline import Pipeline
from steps.finalize import FinalizeStep
from steps.generate_qwen_d11 import GenerateQwenD11Step

pipeline = Pipeline(steps=[
    GenerateQwenD11Step(),
    FinalizeStep()
])
