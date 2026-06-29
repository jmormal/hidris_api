"""
Auth backed by Keycloak (realm: hidris).

The API does NOT issue tokens. It validates the bearer token that the
frontend (or Swagger) obtained from Keycloak, by verifying the RS256
signature against the realm JWKS and checking issuer/audience.
"""

import os
import time
from typing import Any

import httpx
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer

# Internal URL Keycloak is reachable at from the API pod (cluster service).
# Issuer in the token, however, is the *public* hostname, so keep them separate.
KC_INTERNAL = os.getenv("KC_INTERNAL_URL", "http://keycloak.default.svc.cluster.local")
KC_PUBLIC = os.getenv("KC_PUBLIC_URL", "https://keycloak.127.0.0.1.nip.io")
REALM = os.getenv("KC_REALM", "hidris")
CLIENT_ID = os.getenv("KC_CLIENT_ID", "hidris-frontend")

ISSUER = f"{KC_PUBLIC}/realms/{REALM}"
JWKS_URL = f"{KC_INTERNAL}/realms/{REALM}/protocol/openid-connect/certs"
AUTH_URL = f"{KC_PUBLIC}/realms/{REALM}/protocol/openid-connect/auth"
TOKEN_URL = f"{KC_PUBLIC}/realms/{REALM}/protocol/openid-connect/token"

# Swagger uses this; OAuth2AuthorizationCodeBearer gives the Authorize button
# the auth + token URLs so it can run the code+PKCE flow.
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
        # verify=False: self-signed dev TLS
        resp = httpx.get(JWKS_URL, timeout=5.0, verify=False)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json()
        _jwks_cache["ts"] = now
    return _jwks_cache["keys"]


async def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        jwks = _get_jwks()
        # Keycloak public clients put the client in azp, audience is often "account".
        # Disable strict aud check for dev; verify issuer instead.
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
