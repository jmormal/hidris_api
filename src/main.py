"""
Simulation API — ANUGA integration (push-based progress via Redis pub/sub)

    uvicorn main:app --reload --port 8000

Requires a running Redis and at least one RQ worker:
    rq worker jobs:cpu --url redis://localhost:6379

Flow:
  1. POST /api/simulate   -> enqueues run_anuga_job, returns {job_id}
  2. GET  /api/simulate/{job_id}/stream -> SSE driven by a Redis pub/sub
     channel the worker publishes to (no polling for progress)
  3. GET  /api/simulate/{job_id}/result -> returns the SimulationResult JSON
"""

import os
import json
import asyncio

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.registry import (
    FailedJobRegistry,
    FinishedJobRegistry,
    ScheduledJobRegistry,
    StartedJobRegistry,
)

from pydantic import BaseModel
from typing import Any, List, Optional

from src.events import channel_for, decode


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EnqueueResponse(BaseModel):
    message: str
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    status_message: Optional[str] = None
    result: Optional[Any] = None


class JobsByStateResponse(BaseModel):
    queued: List[str]
    started: List[str]
    finished: List[str]
    failed: List[str]
    scheduled: List[str]


# ---------------------------------------------------------------------------
# Redis / Queue
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_QUEUE = os.getenv("REDIS_CPU", "jobs:gpu")

redis_conn = Redis.from_url(REDIS_URL)
q = Queue(REDIS_QUEUE, connection=redis_conn)

JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "3600"))
RESULT_TTL = int(os.getenv("RESULT_TTL", str(60 * 60 * 24)))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs", response_model=JobsByStateResponse, tags=["System"])
async def get_jobs_by_state():
    return {
        "queued": q.get_job_ids(),
        "started": StartedJobRegistry(queue=q).get_job_ids(),
        "finished": FinishedJobRegistry(queue=q).get_job_ids(),
        "failed": FailedJobRegistry(queue=q).get_job_ids(),
        "scheduled": ScheduledJobRegistry(queue=q).get_job_ids(),
    }


def _fetch_job(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Job not found")


# ---------------------------------------------------------------------------
# Submit -> enqueue
# ---------------------------------------------------------------------------


@app.post("/api/simulate", response_model=EnqueueResponse, tags=["Model"])
async def submit_simulation(payload: dict):
    job = q.enqueue(
        "tasks.run_anuga",
        payload,
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        meta={"progress": 0.0, "status_message": "Queued"},
    )
    return {"message": "Simulation enqueued", "job_id": job.id}


@app.get("/job-status/{job_id}", response_model=JobStatusResponse, tags=["Model"])
async def get_job_status(job_id: str):
    """Single-shot status check (the SSE stream is the live path)."""
    job = _fetch_job(job_id)
    status = job.get_status()
    job.refresh()
    progress = job.meta.get("progress", 0)
    status_message = job.meta.get("status_message", "")

    if status == JobStatus.FINISHED:
        return {
            "job_id": job_id,
            "status": status,
            "progress": 100,
            "status_message": "Complete",
            "result": job.result,
        }
    if status == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "status_message": "Job failed",
            "result": job.exc_info,
        }
    return {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "status_message": status_message,
        "result": None,
    }


# ---------------------------------------------------------------------------
# SSE stream <- Redis pub/sub (push, not poll)
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/simulate/{job_id}/stream", tags=["Model"])
async def stream_job(job_id: str):
    """
    Subscribe to the job's Redis channel and forward each published message
    to the client as an SSE event. redis-py pubsub is blocking/sync, so we
    run get_message in a thread and bridge into asyncio via a queue.
    """
    job = _fetch_job(job_id)  # 404 fast on bad id
    channel = channel_for(job_id)

    # Subscribe BEFORE checking terminal state to avoid a race where the job
    # finishes between our status check and our subscription.
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(channel)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()

    def reader():
        # Runs in a worker thread; blocks on Redis, pushes into the asyncio queue.
        while not stop.is_set():
            msg = pubsub.get_message(timeout=1.0)
            if msg and msg.get("type") == "message":
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                loop.call_soon_threadsafe(queue.put_nowait, data)

    async def event_stream():
        reader_task = loop.run_in_executor(None, reader)
        try:
            # If the job already finished/failed before we subscribed, emit the
            # terminal event immediately rather than waiting forever.
            job.refresh()
            status = job.get_status()
            if status == JobStatus.FINISHED:
                yield _sse(
                    "complete",
                    {"job_id": job_id, "file": f"/api/simulate/{job_id}/result"},
                )
                return
            if status == JobStatus.FAILED:
                yield _sse(
                    "error", {"detail": "Simulation failed", "exc": job.exc_info}
                )
                return

            # Optional heartbeat so proxies/clients keep the connection open
            # during long meshing/solve gaps with no progress messages.
            heartbeat = float(os.getenv("SSE_HEARTBEAT", "15"))

            while True:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                payload = decode(raw)
                event = payload.get("event", "message")
                data = payload.get("data", {})
                yield _sse(event, data)

                if event in ("complete", "error"):
                    return
        finally:
            stop.set()
            try:
                await reader_task
            except Exception:
                pass
            pubsub.unsubscribe(channel)
            pubsub.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@app.get("/api/simulate/{job_id}/result", tags=["Model"])
async def get_result(job_id: str):
    job = _fetch_job(job_id)
    await asyncio.sleep(5.5)  # Wait briefly for persistence
    print(job)

    if job.get_status() == JobStatus.FINISHED and job.result is None:
        await asyncio.sleep(0.5)  # Wait briefly for persistence
        job.refresh()
    if job.get_status() != JobStatus.FINISHED:
        raise HTTPException(status_code=409, detail="Job not finished yet")
    result = job.result
    if result is None:
        raise HTTPException(status_code=404, detail="No result available")
    return Response(
        content=result,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )
