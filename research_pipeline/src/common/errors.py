"""Errors raised when pipeline inputs or outputs violate their contract."""


class PipelineError(RuntimeError):
    """A blocking source, split or output validation error."""
