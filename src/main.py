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

import rasterio
from fastapi import FastAPI, HTTPException, Response, Depends, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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
import src.hpc as hpc
from src.remote import RemoteError

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InstanceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None


class InstanceClone(BaseModel):
    # optional override; the API derives "<name> (copy)" when omitted
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)


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
REDIS_QUEUE_CLUSTER_GPU = os.getenv("REDIS_GPU", "jobs:cluster:gpu")
REDIS_QUEUE_CLUSTER_CPU = os.getenv("REDIS_CPU", "jobs:cluster:cpu")

redis_conn = Redis.from_url(REDIS_URL)
q = Queue(REDIS_QUEUE, connection=redis_conn)
q_cluster_gpu = Queue(REDIS_QUEUE_CLUSTER_GPU, connection=redis_conn)

JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "80600"))
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
    expose_headers=["X-Rows", "X-Cols", "X-Min", "X-Max"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


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


@app.exception_handler(RemoteError)
async def _remote_error_handler(request, exc: RemoteError):
    """Anything that failed on the far side of the tailnet is a 502, not a 500.

    Applied app-wide so every HPC route gets it without repeating a try/except:
    the failure is upstream of this API, and the caller should be able to tell
    that apart from a bug in the request.
    """
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health/ssh", tags=["System"])
async def health_ssh(user=Depends(current_user)):
    """Smoke test for the tailnet path to the HPC login node (see src/remote.py)."""
    return {"partitions": await hpc.partitions()}


# ---------------------------------------------------------------------------
# HPC — Slurm job control over SSH (see src/hpc.py)
# ---------------------------------------------------------------------------


@app.post("/api/hpc/probe", tags=["HPC"], status_code=202)
async def submit_hpc_probe(user=Depends(current_user)):
    """
    Queue the connectivity probe (services/hpc-probe) on a compute node.

    Answers whether a compute node can reach this cluster's Redis and Postgres
    over Tailscale, and by which transport — the open question blocking real
    simulations from running there. Returns immediately with a job id; poll
    GET /api/hpc/probe/{job_id} for the verdict.
    """
    job_id = await hpc.submit_probe()
    return {"job_id": job_id, "state": "PENDING"}


@app.get("/api/hpc/probe/{job_id}", tags=["HPC"])
async def get_hpc_probe(job_id: str, log_lines: int = 200, user=Depends(current_user)):
    """
    State of a probe job, plus its report once it has written one.

    `result` stays null while the job is PENDING/RUNNING. `verdict` is lifted
    to the top level as the one field worth reading: which transports reached
    both services, and which of those the unmodified worker db.py could use.
    """
    try:
        state = await hpc.job_state(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = await hpc.probe_result(job_id)
    return {
        "job_id": job_id,
        "state": state,
        "done": state not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"),
        "verdict": (result or {}).get("verdict"),
        "result": result,
        "log": await hpc.probe_log(job_id, lines=log_lines),
    }


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


@app.post("/api/instances/{public_id}/clone", tags=["Model"], status_code=201)
async def clone_instance(
    public_id: UUID, payload: InstanceClone | None = None, user=Depends(current_user)
):
    """Duplicate the setup into a new, unsolved instance. The solution is not copied."""
    row = db.clone_instance(
        user["sub"], str(public_id), name=payload.name if payload else None
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

    Returns raw little-endian float32 bytes (row-major), not JSON — a
    rows*cols float array as JSON text is several times larger and much
    slower for the browser to parse than reading it straight into a
    Float32Array. Shape/range travel as headers since the body is just the
    array. GZipMiddleware compresses the response further (rain grids are
    mostly uniform/zero, so this compresses well).
    """
    # Reads the small precomputed grid — never decompresses the full cube —
    # but still off the event loop since it may fall back to that for a
    # pre-migration row.
    grid, rows, cols, gmin, gmax = await run_in_threadpool(
        db.get_storm_preview_grid, str(public_id)
    )
    if grid is None:
        raise HTTPException(status_code=404, detail="Storm not found")

    return Response(
        content=grid.astype("<f4").tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(rows),
            "X-Cols": str(cols),
            "X-Min": repr(gmin),
            "X-Max": repr(gmax),
        },
    )


# ---------------------------------------------------------------------------
# Result — reads the gzipped solution from the DB (written by the worker)
# ---------------------------------------------------------------------------


@app.get("/api/instances/{public_id}/result", tags=["Model"])
async def get_result(public_id: UUID, user=Depends(current_user)):
    size, chunks, is_solved = db.open_solution(user["sub"], str(public_id))
    if not is_solved or chunks is None:
        raise HTTPException(status_code=409, detail="No solution yet")
    # Binary container (see worker-gpu/src/tasks.py _encode_result_binary),
    # not JSON — the frontend reads it with response.arrayBuffer().
    # Content-Encoding stays gzip so the browser still inflates in transit.
    # Streamed from a Postgres large object in chunks: a solved basin runs to
    # hundreds of MB, so buffering it whole would OOM the API pod.
    # Content-Length describes the gzipped bytes on the wire, which is what
    # we're sending, so the browser still gets a real download progress bar.
    return StreamingResponse(
        chunks,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "Content-Length": str(size),
        },
    )
