"""
SSH to the HPC login node (vrhpcadm1) from the API pod.

The host is reached by its ordinary DNS name (vrhpcadm1.dsic.upv.es), routed
out of the cluster over whatever path the node has to the DSIC network — in
practice the host's UPV VPN. This used to go through the tailscale operator's
egress proxy instead, but vrhpcadm1's own tailscaled has been down since
2026-08, so that ExternalName resolves to a device that no longer answers.
The Service is kept in k8s/api.yaml, unused, for when the node rejoins.

Three auth modes, tried in this order, so a deployment picks whichever
credential it actually has:

  HPC_SSH_KEY       path to a private key  -> normal sshd public-key auth
  HPC_PASSWORD      password               -> normal sshd password auth
  (neither)                                -> keyless, via the Tailscale SSH ACL

The login name is AD-style and contains an "@" (jmormal@upvnet.upv.es), which
is why it is never parsed out of a "user@host" string anywhere in here.
"""

import os
import asyncio

import asyncssh


def _env(*names, default=""):
    """First non-empty of `names`, whitespace-stripped.

    Stripping matters: these come from .env via a Secret, and a trailing space
    on a hostname ("vrhpcadm1.dsic.upv.es ") turns into a DNS failure that
    reads like the host is down.
    """
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


# HPC_SSH_* are the canonical names (config.env / k8s/api.yaml); HPC_SERVER and
# HPC_USERNAME are accepted as aliases because .env carries them under those.
SSH_HOST = _env("HPC_SSH_HOST", "HPC_SERVER", default="vrhpcadm1.dsic.upv.es")
SSH_USER = _env("HPC_SSH_USER", "HPC_USERNAME")
SSH_KEY = _env("HPC_SSH_KEY") or None
SSH_PASSWORD = _env("HPC_SSH_PASSWORD", "HPC_PASSWORD") or None


class RemoteError(RuntimeError):
    """Anything that stopped the command from running to completion.

    Routes map this to a 502 — the failure is on the far side of the tailnet,
    not in the request. asyncssh types never escape this module.
    """


async def run(cmd: str, timeout: float = 30) -> tuple[int, str, str]:
    """Run `cmd` on the login node, returning (exit_status, stdout, stderr).

    A non-zero exit status is returned, not raised: callers like `squeue` care
    about the difference between "the command said no" and "we never got
    there". Only the latter raises.
    """
    if not SSH_USER:
        raise RemoteError("HPC_SSH_USER is unset — see k8s/api.yaml")

    try:
        async with asyncssh.connect(
            SSH_HOST,
            username=SSH_USER,
            # No stable host key to pin: the pod has no known_hosts and the
            # login node's key is not distributed with this repo.
            known_hosts=None,
            # client_keys=None would let asyncssh scan the pod's ~/.ssh; an
            # explicit empty list keeps password auth from being pre-empted by
            # a stray key, and keyless Tailscale SSH needs no key either.
            client_keys=[SSH_KEY] if SSH_KEY else [],
            password=SSH_PASSWORD,
            connect_timeout=timeout,
        ) as conn:
            result = await asyncio.wait_for(
                conn.run(cmd, check=False), timeout=timeout
            )
    except asyncio.TimeoutError:
        raise RemoteError(f"{SSH_HOST}: timed out after {timeout}s running: {cmd}")
    except (OSError, asyncssh.Error) as exc:
        raise RemoteError(f"{SSH_HOST}: {exc}") from exc

    return (
        result.exit_status,
        (result.stdout or "").strip(),
        (result.stderr or "").strip(),
    )


async def check(cmd: str, timeout: float = 30) -> str:
    """`run`, but a non-zero exit status is an error too. Returns stdout."""
    status, out, err = await run(cmd, timeout=timeout)
    if status != 0:
        raise RemoteError(f"{cmd!r} exited {status}: {err or out}")
    return out
