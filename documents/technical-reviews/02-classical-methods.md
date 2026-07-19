# Classical Methods

The classical methods are the established, well-understood ways of choosing features — the kind you'd
reach for today without any reinforcement learning involved. In this study they're the **field to
beat**: if the Decision-Tree-coached learner can't outperform these, the whole idea isn't worth much.
There are four of them, and they represent genuinely different philosophies of what makes a feature
worth keeping.

All four play by the same rules as every other method in the comparison. Each one looks *only* at the
training data when it decides — the validation and test data are never peeked at — and each is scored
afterward by the same shared Decision-Tree yardstick. Each also runs deterministically under the
recorded seed, so the same run always produces the same choices.

## Relevance top-k

The simplest idea: score every feature by how much it tells you about the label on its own, then keep
the best-scoring half. "Tells you about the label" is measured by mutual information — a standard way
of asking how strongly a feature and the answer move together. It's fast and intuitive, but it judges
each feature in isolation, so it can happily keep two features that say almost the same thing.

## Decision-Tree recursive elimination

This method works by *elimination* rather than ranking. It starts with every feature in play, builds a
Decision Tree, and asks which feature the tree leaned on least. It drops that one, rebuilds, and
repeats — peeling away the weakest feature each round — until half the features remain. Because it
re-evaluates the survivors at every step, it can notice that a feature only looked weak because of the
company it was keeping. That thoroughness is also why it's the slowest of the four.

## Minimum redundancy, maximum relevance (mRMR)

This one directly addresses the blind spot of plain relevance ranking. It looks for features that are
both **relevant** to the label *and* **not redundant** with the features already chosen — so instead
of collecting several near-duplicates, it favors a set that covers different ground. It keeps the top
half by this combined criterion. Because results here can differ subtly between software libraries,
the study fixes on one specific, version-pinned implementation and records its identity in the
artifact, so the numbers stay reproducible across machines.

## L1 / LASSO

The odd one out, and the only method that decides its own size. It fits a model with an "L1 penalty" —
a pressure that pushes the influence of unhelpful features all the way down to zero. Whichever features
survive with non-zero influence are the ones selected. How many survive depends on how hard you turn
up that pressure, so unlike the other three (which each keep exactly half the features), L1 might keep
many or few. Features are put on a common scale first, because this kind of penalty is sensitive to
the raw magnitude of the numbers.

## What they share

Three of the four — relevance top-k, Decision-Tree elimination, and mRMR — keep a fixed half of the
features, which makes their subset sizes directly comparable. L1 is the variable-size exception. All
four are single-shot: they make one decision and hand back a subset, with no step-by-step learning.
That's exactly the contrast the reinforced methods are there to test.
