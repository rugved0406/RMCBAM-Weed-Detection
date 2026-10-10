"""WeedVision: RMCBAM-enhanced YOLOv8 weed & crop detection toolkit.

Importing this package registers the RMCBAM classes so that Ultralytics can
unpickle checkpoints trained with them (``weedvision.rmcbam.C2f_RMCBAM``).
"""

from weedvision import rmcbam  # noqa: F401  (registration side effect)

__version__ = "1.0.0"
