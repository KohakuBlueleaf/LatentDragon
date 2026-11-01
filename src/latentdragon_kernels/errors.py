"""Errors raised by the kernel registry and tuner."""


class NoImplementationError(RuntimeError):
    """No backend or candidate can serve the request; the message lists every reason."""
