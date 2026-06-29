"""
Auth backed by Keycloak (realm: hidris).

The API does not issue tokens. It validates the bearer token obtained from
Keycloak by verifying the RS256 signature against the realm JWKS and checking
the issuer.

Two dependencies are exposed:
  - current_user            → reads the Authorization header (normal endpoints)
  - current_user_from_query → reads ?access_token=... (SSE only; EventSource
                              cannot send headers)
Both share _validate_token so the rules can't drift apart.
"""

import os
import time
from typing import Any

import httpx
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, Query
from fastapi.security import OAuth2AuthorizationCodeBearer

KC_INTERNAL = os.getenv("KC_INTERNAL_URL", "http://keycloak.default.svc.cluster.local")
KC_PUBLIC = os.getenv("KC_PUBLIC_URL", "https://keycloak.127.0.0.1.nip.io")
REALM = os.getenv("KC_REALM", "hidris")
CLIENT_ID = os.getenv("KC_CLIENT_ID", "hidris-frontend")

ISSUER = f"{KC_PUBLIC}/realms/{REALM}"
JWKS_URL = f"{KC_INTERNAL}/realms/{REALM}/protocol/openid-connect/certs"
AUTH_URL = f"{KC_PUBLIC}/realms/{REALM}/protocol/openid-connect/auth"
TOKEN_URL = f"{KC_PUBLIC}/realms/{REALM}/protocol/openid-connect/token"

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=AUTH_URL,
    tokenUrl=TOKEN_URL,
    refreshUrl=TOKEN_URL,
)

_jwks_cache: dict[str, Any] = {"keys": None, "ts": 0.0}
_JWKS_TTL = 3600


def _get_jwks() -> dict:
    now = time.time()
    if _jwks_cache["keys"] is None or now - _jwks_cache["ts"] > _JWKS_TTL:
        # self-signed dev TLS
        resp = httpx.get(JWKS_URL, timeout=5.0, verify=False)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json()
        _jwks_cache["ts"] = now
    return _jwks_cache["keys"]


def _validate_token(token: str) -> dict:
    """Single source of truth for token validation."""
    try:
        jwks = _get_jwks()
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            issuer=ISSUER,
            options={"verify_aud": False},
        )
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    return {
        "sub": claims["sub"],
        "username": claims.get("preferred_username"),
        "email": claims.get("email"),
        "roles": claims.get("realm_access", {}).get("roles", []),
        "claims": claims,
    }


async def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return _validate_token(token)


async def current_user_from_query(
    access_token: str = Query(..., description="Keycloak access token (SSE only)"),
) -> dict:
    return _validate_token(access_token)
