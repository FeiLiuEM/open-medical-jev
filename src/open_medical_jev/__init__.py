"""Open Medical Jev — routing-based decision making over frozen open-weight models.

Open Medical Jev runs two untouched, off-the-shelf open models (Qwen3.5-27B and
Qwen3.5-35B-A3B) as *readers* of a yes/no judgment task, and combines their
outputs with a small routing layer: combined confidence, an auto-release gate
(Chow's rule), and a guaranteed candidate set (split conformal prediction).

No fine-tuning. No distillation. No corpus. The recipe is the code in this
package plus the configuration files under ``recipes/``.

License: Apache-2.0. See LICENSE and NOTICE.
"""

__version__ = "0.1.0"

from .router import Router, synth_confidence, conformal_set  # noqa: F401

__all__ = ["Router", "synth_confidence", "conformal_set", "__version__"]
