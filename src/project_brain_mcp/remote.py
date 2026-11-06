"""Serving over Streamable HTTP: bearer-token auth, a health check, and safe defaults for a public host."""

from __future__ import annotations

import hmac
import ipaddress
import json
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

HEALTH_PATH = "/healthz"
MCP_PATH = "/mcp"
MIN_TOKEN_LENGTH = 24


class RemoteConfigError(ValueError):
    """The HTTP settings are unsafe or incomplete."""


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    port: int
    token: str | None
    allowed_hosts: tuple[str, ...] = ()

    @property
    def is_loopback(self) -> bool:
        if self.host == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

    def validate(self) -> None:
        """Refuse to expose the server beyond this machine without a strong token."""
        if self.token is not None and len(self.token) < MIN_TOKEN_LENGTH:
            raise RemoteConfigError(
                f"PROJECT_BRAIN_TOKEN must be at least {MIN_TOKEN_LENGTH} characters; "
                'generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        if not self.is_loopback and not self.token:
            raise RemoteConfigError(
                f"Refusing to listen on {self.host} without PROJECT_BRAIN_TOKEN. Anyone who can reach the port "
                "could use the server. Set a token, or bind to 127.0.0.1."
            )

    def transport_security(self) -> TransportSecuritySettings:
        """DNS-rebinding protection: on for localhost (where it matters most) and for explicit host lists.
        A public, token-protected server behind a platform's router has no fixed Host to check against."""
        if self.allowed_hosts:
            return TransportSecuritySettings(allowed_hosts=list(self.allowed_hosts))
        if self.is_loopback:
            hosts = [f"{name}:{self.port}" for name in ("127.0.0.1", "localhost", "[::1]")]
            return TransportSecuritySettings(allowed_hosts=hosts)
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)


class BearerTokenMiddleware:
    """Requires ``Authorization: Bearer <token>`` on every HTTP request except the health check."""

    def __init__(self, app: ASGIApp, token: str | None):
        self.app = app
        self.token = token.encode() if token else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.token is None or scope["path"] == HEALTH_PATH:
            await self.app(scope, receive, send)
            return
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        scheme, _, presented = header.partition(b" ")
        if scheme.lower() == b"bearer" and hmac.compare_digest(presented.strip(), self.token):
            await self.app(scope, receive, send)
            return
        body = json.dumps({"error": "unauthorized", "message": "Missing or invalid bearer token."}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer realm="project-brain"'),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_app(server: MCPServer, config: RemoteConfig) -> ASGIApp:
    config.validate()

    @server.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = server.streamable_http_app(
        streamable_http_path=MCP_PATH, transport_security=config.transport_security(), host=config.host
    )
    return BearerTokenMiddleware(app, config.token)


def serve(server: MCPServer, config: RemoteConfig) -> None:
    import uvicorn

    uvicorn.run(build_app(server, config), host=config.host, port=config.port, log_level="info")
