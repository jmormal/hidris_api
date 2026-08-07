"""
Slurm job control on the HPC cluster, over the SSH transport in src/remote.py.

Scope is deliberately narrow: submit the connectivity probe, ask what happened
to it, read its JSON report back. This is the thin end of the wedge for running
real simulations on the cluster, but nothing here assumes that yet.

Every command is built from a validated job id — never from raw user input.
Commands run through a login shell on the far side, so an unvalidated id would
be a command injection with the API's SSH identity.
"""

import json
import re

from src.remote import RemoteError, check, run

# Layout on the cluster, matching services/hpc-probe/run-probe.slurm.
# $HOME rather than ~: these paths get interpolated into shell commands, and a
# tilde does not expand inside quotes. $HOME does, and survives either way.
CONTAINERS_DIR = "$HOME/containers"
PROBE_SCRIPT = "run-probe.slurm"  # relative: submitted from CONTAINERS_DIR

# Slurm job ids are numeric, optionally with an array-task suffix (12345_7).
# \Z rather than $: $ also matches before a trailing newline, which would let
# "12345\n" through into a shell command string.
_JOB_ID_RE = re.compile(r"^\d+(_\d+)?\Z")


def _validate_job_id(job_id: str) -> str:
    """Reject anything that is not a Slurm job id, before it reaches a shell."""
    if not _JOB_ID_RE.match(job_id):
        raise ValueError(f"not a valid Slurm job id: {job_id!r}")
    return job_id


async def submit_probe() -> str:
    """sbatch the probe, returning its job id.

    --parsable makes sbatch print just the id (or "id;cluster"), which is far
    more robust than parsing "Submitted batch job 12345".
    """
    # Submitted from the containers dir, not $HOME: Slurm resolves --output
    # (%x-%j.out) relative to the submission directory, so this is what makes
    # the .out file land somewhere probe_log can predict.
    out = await check(
        f"cd {CONTAINERS_DIR} && sbatch --parsable {PROBE_SCRIPT}", timeout=60
    )
    job_id = out.strip().split(";")[0].strip()
    if not _JOB_ID_RE.match(job_id):
        raise RemoteError(f"sbatch returned an unparseable job id: {out!r}")
    return job_id


async def job_state(job_id: str) -> str:
    """Current Slurm state: PENDING, RUNNING, COMPLETED, FAILED, ...

    squeue only knows about jobs still in the queue; once a job leaves, it has
    to come from sacct. Trying squeue first is the cheaper of the two and is
    the path taken while the caller is actually polling.
    """
    _validate_job_id(job_id)

    status, out, _ = await run(f"squeue -j {job_id} -h -o %T", timeout=30)
    if status == 0 and out.strip():
        return out.strip().splitlines()[0].strip()

    # -X collapses the batch/extern steps into the job's own row.
    status, out, _ = await run(f"sacct -j {job_id} -n -X -o State%20", timeout=30)
    if status == 0 and out.strip():
        # "CANCELLED by 12345" -> "CANCELLED"
        return out.strip().splitlines()[0].strip().split()[0]

    # slurmdbd is down on this cluster (sacct fails with "Connection refused" on
    # 6819), so accounting cannot answer once a job has left the queue. Rather
    # than failing the request, infer the state from what the job left on disk —
    # which is what the caller actually wants to know anyway.
    return await _state_from_artifacts(job_id)


async def _state_from_artifacts(job_id: str) -> str:
    """Best-effort state for a job squeue has forgotten and sacct cannot reach.

    Deliberately coarse: the only distinctions that matter to a caller polling
    for results are "there is a report", "it ran but produced nothing", and "no
    trace of it".
    """
    result = f"{CONTAINERS_DIR}/probe-{job_id}/probe-result.json"
    log = f"{CONTAINERS_DIR}/hidris-probe-{job_id}.out"
    _, out, _ = await run(
        f'if [ -s "{result}" ]; then echo COMPLETED; '
        f'elif [ -f "{log}" ]; then echo FAILED; '
        f"else echo UNKNOWN; fi",
        timeout=30,
    )
    return out.strip() or "UNKNOWN"


async def probe_result(job_id: str) -> dict | None:
    """The merged JSON report, or None if the job has not written one yet.

    A missing file is the normal state for a queued or running job, so it is
    not an error — the caller distinguishes using job_state.
    """
    _validate_job_id(job_id)
    path = f"{CONTAINERS_DIR}/probe-{job_id}/probe-result.json"

    # Unquoted so $HOME expands; safe because job_id is validated to digits
    # above, so the interpolation cannot introduce shell metacharacters.
    status, out, _ = await run(f'cat "{path}"', timeout=60)
    if status != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"probe report for {job_id} is not valid JSON: {exc}")


async def probe_log(job_id: str, lines: int = 200) -> str:
    """Tail of the Slurm .out file — the human-readable side of the probe.

    Worth surfacing even on success, and it is the only diagnostic available
    when the job dies before writing any JSON at all.
    """
    _validate_job_id(job_id)
    # %x-%j.out relative to the submission directory, which submit_probe pins
    # to CONTAINERS_DIR. Keep the two in step if either moves.
    path = f"{CONTAINERS_DIR}/hidris-probe-{job_id}.out"
    status, out, _ = await run(
        f"tail -n {int(lines)} {path} 2>/dev/null", timeout=30
    )
    return out if status == 0 else ""


async def partitions() -> str:
    """`sinfo -s` — also serves as the cheapest possible liveness check."""
    return await check("sinfo -s", timeout=30)
