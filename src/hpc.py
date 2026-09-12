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
SIM_SCRIPT = "run-simulation.slurm"
# Separate script, not a flag: the multi-node path launches one container per
# task with srun --mpi=pmix and gives only rank 0 a tailnet, where the
# single-node path runs one container that starts mpirun itself inside a network
# namespace shared by every rank. The two launch models have almost nothing in
# common, so they stay apart rather than growing conditionals.
SIM_SCRIPT_MN = "run-simulation-mn.slurm"
MAX_NODES = 6  # the cluster has six

# Slurm job ids are numeric, optionally with an array-task suffix (12345_7).
# \Z rather than $: $ also matches before a trailing newline, which would let
# "12345\n" through into a shell command string.
_JOB_ID_RE = re.compile(r"^\d+(_\d+)?\Z")

# Anything interpolated into a remote shell command must match one of these.
_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}\Z")


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


# Bounds for the sbatch resource overrides. These are a guard against a bad
# request reaching a shared scheduler, not a statement about the hardware: the
# nodes have 8 GPUs and 490GB each, and a job asking for more than exists sits
# in PENDING forever rather than failing, which is a confusing way to find out.
MAX_GPUS = 8
MAX_MEM_GB = 480
MIN_MEM_GB = 4


def _validate_resources(
    gpus: int | None, mem_gb: int | None, nodes: int | None = None
) -> tuple[str, str]:
    """Turn optional gpu/memory requests into sbatch flags.

    Returns ("", "") when neither is set, so the #SBATCH directives baked into
    run-simulation.slurm stay in force — the script remains runnable by hand.
    """
    gres_flag = ""
    if gpus is not None:
        if not isinstance(gpus, int) or not 1 <= gpus <= MAX_GPUS:
            raise ValueError(f"gpus must be 1-{MAX_GPUS}, got {gpus!r}")
        # CPUs must scale with ranks. run-simulation.slurm hardcodes
        # --cpus-per-task=4, so asking for 5 GPUs used to give 5 MPI ranks
        # 4 cores to share — genuine oversubscription during the (CPU-bound)
        # mesh and partition phases. One core per rank plus headroom for the
        # host-side RK loop.
        gres_flag = f"--gres=gpu:{gpus} --cpus-per-task={max(4, gpus + 1)} "

    if nodes is not None:
        if not isinstance(nodes, int) or not 1 <= nodes <= MAX_NODES:
            raise ValueError(f"nodes must be 1-{MAX_NODES}, got {nodes!r}")
        if nodes > 1:
            # gpus is PER NODE here — --gres always is — so one rank per GPU on
            # each node gives nodes*gpus ranks in total.
            per_node = gpus or 1
            gres_flag += f"--nodes={nodes} --ntasks-per-node={per_node} "

    mem_flag = ""
    if mem_gb is not None:
        if not isinstance(mem_gb, int) or not MIN_MEM_GB <= mem_gb <= MAX_MEM_GB:
            raise ValueError(
                f"mem_gb must be {MIN_MEM_GB}-{MAX_MEM_GB}, got {mem_gb!r}"
            )
        mem_flag = f"--mem={mem_gb}G "

    return gres_flag, mem_flag


async def submit_simulation(
    public_id: str,
    stream_job_id: str,
    gpus: int | None = None,
    mem_gb: int | None = None,
    nodes: int | None = None,
) -> str:
    """sbatch one ANUGA simulation, returning the Slurm job id.

    `stream_job_id` is the id the frontend will open the SSE stream on; the
    worker publishes to sim:events:{stream_job_id}. It is generated by the API
    rather than derived from the Slurm id because the client needs it in the
    response to the submit call, before Slurm has assigned anything.

    `gpus` and `mem_gb` override the #SBATCH directives in the script. Command
    line beats in-script directives in sbatch, so the script keeps working
    unchanged when submitted by hand. Both are integer-validated before they
    reach the remote shell, same as the ids.

    Both ids are UUID-validated: they are interpolated into a command that runs
    in a login shell on the far side, with the API's SSH identity.
    """
    for name, value in (("public_id", public_id), ("job_id", stream_job_id)):
        if not _UUID_RE.match(value):
            raise ValueError(f"{name} is not a UUID: {value!r}")

    gres_flag, mem_flag = _validate_resources(gpus, mem_gb, nodes)
    script = SIM_SCRIPT_MN if (nodes or 1) > 1 else SIM_SCRIPT

    # --export=ALL,... keeps the login environment and adds ours on top; the
    # slurm script requires both and fails fast without them.
    out = await check(
        f"cd {CONTAINERS_DIR} && sbatch --parsable "
        f"{gres_flag}{mem_flag}"
        f"--export=ALL,PUBLIC_ID={public_id},JOB_ID={stream_job_id} "
        f"{script}",
        timeout=60,
    )
    slurm_id = out.strip().split(";")[0].strip()
    if not _JOB_ID_RE.match(slurm_id):
        raise RemoteError(f"sbatch returned an unparseable job id: {out!r}")
    return slurm_id


async def simulation_log(slurm_job_id: str, lines: int = 200) -> str:
    """Tail of a simulation job's Slurm output."""
    _validate_job_id(slurm_job_id)
    path = f"{CONTAINERS_DIR}/hidris-sim-{slurm_job_id}.out"
    status, out, _ = await run(f'tail -n {int(lines)} "{path}" 2>/dev/null', timeout=30)
    return out if status == 0 else ""


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

    Both Slurm scripts end by printing `=== exit N ===`, which is a more
    reliable completion signal than the presence of any output file — a job
    still running has a log but no exit line. Covers probe and simulation jobs
    alike, since they share that convention.
    """
    # %x-%j.out, and %x differs per script, so glob the job id across both.
    logs = f'"{CONTAINERS_DIR}"/hidris-*-{job_id}.out'
    _, out, _ = await run(
        f'f=$(ls {logs} 2>/dev/null | head -1); '
        f'if [ -z "$f" ]; then echo UNKNOWN; '
        f'elif grep -q "=== exit 0 ===" "$f"; then echo COMPLETED; '
        f'elif grep -q "=== exit " "$f"; then echo FAILED; '
        f"else echo RUNNING; fi",
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
