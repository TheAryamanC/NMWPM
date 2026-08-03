"""nmwpm — Neural Minimum-Weight Perfect Matching for quantum error correction.

Quickstart
----------
Load a pretrained decoder and decode a syndrome in three lines::

    import nmwpm
    decoder = nmwpm.Decoder.from_checkpoint("checkpoints/nmwpm_toric_L8_depolarizing.pt")
    parity  = decoder.decode(syndrome)   # shape: (2 * num_logicals,)

See the project README for full documentation and examples.
"""

from .codes import CSSCode, ToricCode, RotatedSurfaceCode
from .model import QWP, build_batch
from .decoder import Decoder

__all__ = [
    "CSSCode",
    "ToricCode",
    "RotatedSurfaceCode",
    "QWP",
    "build_batch",
    "Decoder",
]

__version__ = "0.1.0"
