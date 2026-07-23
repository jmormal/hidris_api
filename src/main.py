"""
Simulation API — ANUGA integration (push-based progress via Redis pub/sub)

    uvicorn main:app --reload --port 8000

Requires a running Redis and at least one RQ worker:
    rq worker jobs:cpu --url redis://localhost:6379

Auth: tokens are issued by Keycloak (realm: hidris). This API only validates
them. The SSE stream takes the token as ?access_token= because EventSource
cannot send an Authorization header.
"""

import os
import json
import asyncio
import contextlib
from datetime import date

from uuid import UUID

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Response, Depends, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

from pydantic import BaseModel, Field
from typing import Any, List, Optional

from src.auth import current_user, current_user_from_query, CLIENT_ID
import src.db as db
from src.events import channel_for, decode


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InstanceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None


class InstanceUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = None
    # serialized setup payload (features + config)
    instance: Optional[dict] = None


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

app = FastAPI(
    swagger_ui_init_oauth={
        "clientId": CLIENT_ID,
        "usePkceWithAuthorizationCodeGrant": True,
        "scopes": "openid profile email",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Idempotent; ensures the tables exist before the first request.
    db.init_db()
    db.init_storms()


# ---------------------------------------------------------------------------
# Auth / health
# ---------------------------------------------------------------------------


@app.get("/api/me", tags=["Auth"])
async def me(user=Depends(current_user)):
    return user


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Jobs (system)
# ---------------------------------------------------------------------------


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
# Instances (CRUD)
# ---------------------------------------------------------------------------


@app.get("/api/instances", tags=["Model"])
async def list_instances(user=Depends(current_user)):
    return db.list_instances(user["sub"])


@app.post("/api/instances", tags=["Model"], status_code=201)
async def create_instance(payload: InstanceCreate, user=Depends(current_user)):
    return db.create_instance(user["sub"], payload.name, payload.description)


@app.get("/api/instances/{public_id}", tags=["Model"])
async def get_instance(public_id: UUID, user=Depends(current_user)):
    row = db.get_instance(user["sub"], str(public_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Instance not found")
    return row


@app.patch("/api/instances/{public_id}", tags=["Model"])
async def update_instance(
    public_id: UUID, payload: InstanceUpdate, user=Depends(current_user)
):
    row = db.update_instance(
        user["sub"],
        str(public_id),
        name=payload.name,
        description=payload.description,
        instance=payload.instance,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Instance not found")
    return row


@app.delete("/api/instances/{public_id}", tags=["Model"], status_code=204)
async def delete_instance(public_id: UUID, user=Depends(current_user)):
    if not db.delete_instance(user["sub"], str(public_id)):
        raise HTTPException(status_code=404, detail="Instance not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Submit -> enqueue (per instance)
# ---------------------------------------------------------------------------


@app.post(
    "/api/instances/{public_id}/simulate",
    response_model=EnqueueResponse,
    tags=["Model"],
)
async def submit_simulation(public_id: UUID, user=Depends(current_user)):
    inst = db.get_instance(user["sub"], str(public_id))
    if inst is None:
        raise HTTPException(status_code=404, detail="Instance not found")

    job = q.enqueue(
        "tasks.run_anuga",
        {"public_id": str(public_id), "payload": inst["instance"]},
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        meta={"progress": 0.0, "status_message": "Queued"},
    )
    return {"message": "Simulation enqueued", "job_id": job.id}


@app.get("/job-status/{job_id}", response_model=JobStatusResponse, tags=["Model"])
async def get_job_status(job_id: str, user=Depends(current_user)):
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
            "result": None,
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
# Auth via ?access_token= because EventSource can't set headers.
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/simulate/{job_id}/stream", tags=["Model"])
async def stream_job(job_id: str, user=Depends(current_user_from_query)):
    job = _fetch_job(job_id)  # 404 fast on bad id
    channel = channel_for(job_id)

    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(channel)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()

    def reader():
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
            job.refresh()
            status = job.get_status()
            if status == JobStatus.FINISHED:
                yield _sse("complete", {"job_id": job_id})
                return
            if status == JobStatus.FAILED:
                yield _sse(
                    "error", {"detail": "Simulation failed", "exc": job.exc_info}
                )
                return

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
# Storms — historical rainfall catalog (shared reference data, not per-user)
# ---------------------------------------------------------------------------


@app.get("/api/storms", tags=["Storms"])
async def list_storms(user=Depends(current_user)):
    return db.list_storms()


@app.get("/api/storms/{public_id}", tags=["Storms"])
async def get_storm(public_id: UUID, user=Depends(current_user)):
    row = db.get_storm_meta(str(public_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Storm not found")
    return row


@app.post("/api/storms", tags=["Storms"], status_code=201)
async def create_storm(
    files: List[UploadFile] = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    event_date: Optional[str] = Form(None),
    timestep_s: int = Form(...),
    units: str = Form("mm_per_step"),
    user=Depends(current_user),
):
    """
    Frames are stacked in the order `files` arrives in the multipart body —
    the client is responsible for ordering them (see AddStormModal).
    """
    if units not in ("mm_per_step", "mm_hr"):
        raise HTTPException(
            status_code=400, detail="units must be mm_per_step or mm_hr"
        )
    if not files:
        raise HTTPException(status_code=400, detail="At least one raster is required")

    with contextlib.ExitStack() as stack:
        readers = []
        for f in files:
            data = await f.read()
            memfile = stack.enter_context(rasterio.MemoryFile(data))
            readers.append(stack.enter_context(memfile.open()))
        try:
            # Stacking + normalising every frame is CPU-bound (and, for
            # real-resolution rasters, not fast) — running it inline would
            # block the whole event loop for the duration of the upload.
            cube, cell_size_m, nodata, center_lng, center_lat = await run_in_threadpool(
                db.build_cube_from_rasters, readers
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    ev = date.fromisoformat(event_date) if event_date else None

    # gzip-compressing the cube and writing the blob is also CPU-bound.
    public_id = await run_in_threadpool(
        db.insert_storm,
        name=name,
        description=description,
        event_date=ev,
        timestep_s=timestep_s,
        cube=cube,
        cell_size_m=cell_size_m,
        units=units,
        nodata=nodata,
        center_lng=center_lng,
        center_lat=center_lat,
    )
    return {"public_id": public_id}


@app.get("/api/storms/{public_id}/preview", tags=["Storms"])
async def get_storm_preview(public_id: UUID, user=Depends(current_user)):
    """
    Accumulated rainfall (mm) per cell, summed across every frame — the
    bitmap shown when a storm is placed on the map.
    """
    # Decompressing the cube (gzip) is CPU-bound — keep it off the event loop.
    cube, meta = await run_in_threadpool(db.get_storm_cube, str(public_id))
    if cube is None:
        raise HTTPException(status_code=404, detail="Storm not found")

    timestep_s = float(meta["timestep_s"])
    if meta["units"] == "mm_hr":
        accum = (cube * (timestep_s / 3600.0)).sum(axis=0)
    else:
        accum = cube.sum(axis=0)
    accum = np.nan_to_num(accum, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "rows": int(accum.shape[0]),
        "cols": int(accum.shape[1]),
        "units": "mm",
        "values": accum.astype(float).flatten().tolist(),
        "min": float(accum.min()),
        "max": float(accum.max()),
    }


# ---------------------------------------------------------------------------
# Result — reads the gzipped solution from the DB (written by the worker)
# ---------------------------------------------------------------------------


@app.get("/api/instances/{public_id}/result", tags=["Model"])
async def get_result(public_id: UUID, user=Depends(current_user)):
    raw, is_solved = db.get_solution_bytes(user["sub"], str(public_id))
    if not is_solved or raw is None:
        raise HTTPException(status_code=409, detail="No solution yet")
    return Response(
        content=raw,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )
