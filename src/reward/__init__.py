"""Personalized reward (BLOCK-003).

Owns the reward signal — an accuracy-minus-correlation overall reward (COMP-008) personalized per
agent (COMP-009) — reusing the correlation structure the state block (:mod:`state.graph`) already
builds rather than recomputing it (DEC-003). The overall scalar lives in :mod:`reward.overall`; per-
agent personalization is layered on in TASK-404.
"""
