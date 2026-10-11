"""Public interface for ResilientFile."""

from .encoder import encode
from .error_decoder import DecodeResult, decode
from .erasure_decoder import RecoveryError, recover as recover_erasures
from .container import ContainerError, protect_bytes, recover_bytes

__version__ = "0.1.0"

__all__ = [
    "encode",
    "decode",
    "recover_erasures",
    "protect_bytes",
    "recover_bytes",
    "DecodeResult",
    "RecoveryError",
    "ContainerError",
    "__version__",
]