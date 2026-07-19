"""Engine package — the no-trainer reinforced selection engine and its internal seam.

Holds the engine's *internal* state/reward seam (REQ-014): the one interface through
which the exploration loop reads each agent's fixed-length state vector and receives its
per-step reward. The provisional minimal state (TASK-206) and overall reward (TASK-207)
are built *against* this seam now; PHASE-004's Decision-Tree-structured state and
personalized per-agent reward replace them behind it with no change to the engine, the
agents, or the common subset contract (RISK-004, verified against a stand-in at TASK-212).

This seam is internal to the engine and is deliberately distinct from
``harness.contract`` (the external equal-footing ``Selector`` contract): the engine
*satisfies* that subset contract at its boundary (TASK-210) while *consuming* its state
and reward through this separate, replaceable interface.
"""
