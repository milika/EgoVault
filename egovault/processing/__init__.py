"""EgoVault processing layer — retrieval and enrichment pipelines.

During the incremental migration (Cycle 7.5) the actual RAG code lives in
``egovault.chat.rag``.  This package re-exports everything from there so that
code written against the new import path works immediately.

Once all callers import from ``egovault.processing.rag`` and tests are updated
to patch ``egovault.processing.rag.*``, the actual code will move here and
``chat/rag.py`` will become the shim.  Until then, this is the shim.
"""
