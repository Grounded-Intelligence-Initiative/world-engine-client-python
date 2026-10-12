"""Client-side exceptions mirroring the server's error codes.

Server replies carry ``{ok:false, code, error, fixes:[...]}``; ``raise_for``
maps codes to typed exceptions and renders the fixes into the message the same
way the World Engine CLI does ("Possible fixes:" block).
"""

from __future__ import annotations


class WorldEngineClientError(RuntimeError):
    """Base error; carries the server error code when one exists."""

    def __init__(self, message: str, *, code: str | None = None, fixes: list[str] | None = None):
        if fixes:
            message += "\n  Possible fixes:\n" + "\n".join(
                f"    {i + 1}. {f}" for i, f in enumerate(fixes)
            )
        super().__init__(message)
        self.code = code


class TokenError(WorldEngineClientError):
    """The control token was rejected (wrong token, or the server was started
    without --allow-remote-control)."""


class ControllerConflictError(WorldEngineClientError):
    """Another controller is bound; pass takeover=True to seize control."""


class ResetRequiredError(WorldEngineClientError):
    """step() outside an active episode — call reset() first."""


class ProtocolError(WorldEngineClientError):
    """The server said something this client version does not understand."""


class TeleopActiveError(WorldEngineClientError):
    """Teleop is enabled in the editor — the human at the browser owns the
    environment. Ask them to disable teleop in the Inspector (or wait)."""


class BadConfigError(WorldEngineClientError):
    """configure()/connect() asked for an unknown control_mode or obs key —
    see env.metadata['supported_control_modes'] / ['supported_obs_keys']."""


class RecoveryRequiredError(WorldEngineClientError):
    """An action may have executed; explicit reset/reconnection is required."""


class RecordingPendingError(WorldEngineClientError):
    """Save or discard recording data in the editor before resetting."""


_BY_CODE = {
    "recovery_required": RecoveryRequiredError,
    "recording_pending": RecordingPendingError,
    "unsupported_protocol": ProtocolError,
    "instance_mismatch": ProtocolError,
    "bad_token": TokenError,
    "controller_exists": ControllerConflictError,
    "reset_required": ResetRequiredError,
    "episode_changed": ResetRequiredError,
    "bad_session": ControllerConflictError,
    "not_connected": ProtocolError,
    "teleop_active": TeleopActiveError,
    "bad_config": BadConfigError,
}


def raise_for(reply: dict) -> None:
    """Raise the typed exception for a ``{ok:false}`` server reply."""
    code = reply.get("code")
    cls = _BY_CODE.get(code, WorldEngineClientError)
    raise cls(reply.get("error", "unknown server error"), code=code, fixes=reply.get("fixes"))
