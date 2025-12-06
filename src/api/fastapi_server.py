"""
FastAPI Server for ML Conductor
Production-ready server with async support, automatic documentation, and proper error handling
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from async_worker.async_worker import AsyncWorker
from async_worker.s3_pipeline_client import S3PipelineClient


# Pydantic Models for request/response validation
class ReferenceImage(BaseModel):
    """Reference image with S3 ID and influence strength"""
    imageId: str = Field(..., description="S3 image ID")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Influence strength (0.0-1.0)")


class GenerateImageRequest(BaseModel):
    """Request model for image generation"""
    userId: str = Field(..., description="User ID for S3 path structure")
    prompt: str = Field(..., description="Text prompt for image generation")
    negativePrompt: Optional[str] = Field(None, description="Negative prompt to avoid certain features")
    width: int = Field(512, ge=256, le=2048, description="Image width in pixels")
    height: int = Field(768, ge=256, le=2048, description="Image height in pixels")
    steps: int = Field(30, ge=1, le=150, description="Number of inference steps")
    guidanceScale: float = Field(7.5, ge=1.0, le=30.0, description="Guidance scale for prompt adherence")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Reference images for composition - each with imageId + strength
    characterReferences: Optional[list[ReferenceImage]] = Field(None, description="Character reference images (min 2 for multi-character scenes)")
    settingReference: Optional[ReferenceImage] = Field(None, description="Setting/environment/stage reference image")
    objectReferences: Optional[list[ReferenceImage]] = Field(None, description="Object/prop reference images")


class GenerateImageResponse(BaseModel):
    """Response model for image generation"""
    jobId: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, processing, completed, failed")
    imageUrl: Optional[str] = Field(None, description="URL of generated image (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    jobId: str
    status: str
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    imageUrl: Optional[str] = None
    error: Optional[str] = None


# Global job lock for sequential processing
job_semaphore = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    print("🚀 Starting Conductor ML Server...")
    print("📚 API Documentation available at /docs")
    yield
    # Shutdown
    print("👋 Shutting down Conductor ML Server...")


# Initialize FastAPI app
app = FastAPI(
    title="Conductor ML Server",
    description="Production-ready ML server for image generation with Stable Diffusion",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - DISABLED FOR TESTING
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# S3 Client
s3_client = S3PipelineClient()


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "conductor-ml"}


@app.post("/api/generate", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate an image using Stable Diffusion
    
    This endpoint queues an image generation job and returns immediately.
    Use the returned jobId to check status via /api/jobs/{jobId}
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    user_id = request.userId
    
    # Prepare input data
    input_data = {
        "user_id": user_id,
        # Map to pipeline-expected parameter names
        "prompt_positive": request.prompt,
        "prompt_negative": request.negativePrompt or "",
        "width": request.width,
        "height": request.height,
        "inference_steps": request.steps,
        "ai_creativity": request.guidanceScale,
        "seed": request.seed,
        "character_references": [ref.dict() for ref in request.characterReferences] if request.characterReferences else None,
        "setting_reference": request.settingReference.dict() if request.settingReference else None,
        "object_references": [ref.dict() for ref in request.objectReferences] if request.objectReferences else None,
    }
    
    # Save initial status to S3
    s3_client.write_json(user_id, job_id, "request.json", {"input": input_data})
    s3_client.write_json(user_id, job_id, "status.json", {
        "status": "queued",
        "job_id": job_id,
    })
    
    # Queue background task
    background_tasks.add_task(run_generation_job, user_id, job_id, input_data)
    
    return GenerateImageResponse(
        jobId=job_id,
        status="queued",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, user_id: str = "default"):
    """
    Get the status of a generation job
    
    Returns the current status and result URL when completed
    """
    status_data = s3_client.read_json(user_id, job_id, "status.json")
    
    if status_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check for result
    result_data = s3_client.read_json(user_id, job_id, "result.json")
    image_url = None
    if result_data and "output_url" in result_data:
        image_url = result_data["output_url"]
    
    return JobStatusResponse(
        jobId=job_id,
        status=status_data.get("status", "unknown"),
        progress=status_data.get("progress"),
        imageUrl=image_url,
        error=status_data.get("error"),
    )


async def run_generation_job(user_id: str, job_id: str, input_data: dict):
    """
    Run the actual generation job in background
    Uses semaphore to ensure only one job runs at a time
    """
    async with job_semaphore:
        try:
            # Update status to processing
            s3_client.write_json(user_id, job_id, "status.json", {
                "status": "processing",
                "job_id": job_id,
            })
            
            # Run worker (this is synchronous, so we run in executor)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                _run_worker_sync,
                user_id,
                job_id,
                input_data,
            )
            
        except Exception as e:
            # Update status to failed
            s3_client.write_json(user_id, job_id, "status.json", {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
            })


def _run_worker_sync(user_id: str, job_id: str, input_data: dict):
    """Synchronous worker wrapper for running in executor"""
    worker = AsyncWorker(user_id, job_id, input_data)
    worker.run()


# Mount static files if available (for UI)
try:
    app.mount("/ui", StaticFiles(directory="static/ui", html=True), name="ui")
except RuntimeError:
    # Static directory doesn't exist yet
    pass
