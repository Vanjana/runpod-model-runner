"""
Conductor ML Server - FastAPI Entry Point
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from api.fastapi_server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
