# Simple RL

The classical methods each make a single decision and hand back a subset. Reinforcement learning
takes a different stance: treat feature selection as something you get better at by **trying,
scoring, and adjusting** — over and over — rather than deciding once. In the current implementation,
this plain, uncoached control is named **`marlfs`**. It uses the minimal state and one uniform reward
for every agent, with no coaching, tree-structured state, or personalized reward.

## How it explores

Instead of one selector making one choice, imagine giving **each feature its own tiny decision-maker**
— an agent whose only job is to vote on whether *its* feature should be in the subset. The run starts
from a random half of the features and then proceeds in a fixed number of rounds (a "budget" of
steps). Each round:

1. Every agent looks at the current situation and votes **keep** or **drop** for its feature.
2. Mostly an agent votes for whatever has worked best so far, but every so often it tries the other
   option instead — a deliberate dash of experimentation so the search doesn't get stuck on its first
   guess.
3. The features voted "keep" form the new subset, which is scored once by the shared Decision-Tree
   yardstick, producing an accuracy for that round.
4. That result is turned into a **reward**, and each agent nudges its future voting toward whatever
   earns a higher reward. This is the "learning" part — the agents' judgment improves as the rounds go
   by.

Two safeguards keep the search honest. A round that would select *everything* or *nothing* is simply
rejected — the subset stays strictly in between — so the learner can't cheat by grabbing all the
features or collapse to an empty set. And throughout, the run keeps a record of the accuracy at every
step and remembers the **best subset it ever saw**, which is what it ultimately returns.

## Why it's "simple"

What makes this the *plain* version is what's deliberately missing: nothing is telling the agents
which features the evidence actually favors. Every agent gets the **same single reward signal**, and
each works from a **minimal picture** of the situation. It learns purely from trial and error on the
scores it happens to see.

That's the whole point of having this arm. It provides the uncoached control that the richer IRFS
variants must beat. The run uses a fixed exploration budget and returns the best subset it observed;
it records the trajectory, but does not declare a separate mathematical convergence test. Simple RL is
the foundation; **full IRFS** is what you get when tree-derived state, personalized reward, and hybrid
teaching are added.
