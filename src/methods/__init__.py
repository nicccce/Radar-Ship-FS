"""Selection methods — concrete selectors satisfying the common subset contract.

Holds the classical baselines (this thin slice: relevance top-k) that conform to
``harness.contract.Selector`` and are invoked by the orchestrator through that one interface. The
no-trainer reinforced engine (inner PHASE-002) lives separately and satisfies the same contract at
its boundary.
"""
