"""EgoVault agent core layer.

During the incremental migration (Cycle 7.5) the canonical implementations
live in ``egovault.chat.session``.  This package exposes the new public API
(``AgentSession``) and re-exports every existing symbol so that code can be
gradually updated to import from ``egovault.agent.*``.

Once all callers and tests are updated, the code will physically move here.
"""
from egovault.agent.session import AgentSession  # noqa: F401
