"""
Simulation API — test stub

    uvicorn main:app --reload --port 8000

To return a real result file, either:
  1. Set RESULT_FILE below to a .json path
  2. Or drop a file named "result.json" next to this script
"""

import asyncio
import uuid
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for submitted payloads
jobs: dict[str, dict] = {}

# Point this at your flood viewer .json to auto-load results
# e.g. RESULT_FILE = Path("output.json")
RESULT_FILE: Path | None = Path("result.json") if Path("result.json").exists() else None

RESULT_FILE = Path(__file__).parent / "result.json"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/simulate")
async def submit_simulation(payload: dict):
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = payload
    print(f"\n{'=' * 50}")
    print(f"JOB {job_id} received")
    print(json.dumps(payload, indent=2, default=str))
    print(f"{'=' * 50}\n")
    with open("test.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return {"job_id": job_id}


@app.get("/api/simulate/{job_id}/stream")
async def stream_job(job_id: str):
    async def event_stream():
        # 1. Meshing
        yield f"event: meshing\ndata: {{}}\n\n"
        await asyncio.sleep(2)

        # 2. Solving progress
        for pct in (20, 50, 80, 100):
            yield f"event: progress\ndata: {json.dumps({'progress': pct})}\n\n"
            await asyncio.sleep(0.75)

        # 3. Complete — include file URL if a result exists
        result: dict = {"job_id": job_id}
        if RESULT_FILE and RESULT_FILE.exists():
            result["file"] = f"/api/simulate/{job_id}/result"

        yield f"event: complete\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/simulate/{job_id}/result")
async def get_result(job_id: str):
    """Return the result .json file for the visualizer to consume."""
    if RESULT_FILE and RESULT_FILE.exists():
        return FileResponse(
            RESULT_FILE,
            media_type="application/json",
            filename=f"{job_id}.json",
        )
    return {"error": "No result file. Place a result.json next to main.py."}
