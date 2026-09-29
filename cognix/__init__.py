"""cognix: a cognitive architecture runtime.

Layered memory (working, episodic, semantic), salience-driven attention,
a planner-executor agent loop with reflection, real tools, evals, and
persistence. Standard library only.
"""

__version__ = "1.0.0"

from .runtime import CognitiveRuntime

__all__ = ["CognitiveRuntime", "__version__"]
