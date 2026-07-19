# Full IRFS

Full IRFS is simple RL with a **Decision Tree put "in the loop"** — the method the whole study exists
to test. It keeps the exploration machinery of [simple RL](03-simple-rl.md) intact — the per-feature
agents, the budget of steps, the vote-score-adjust cycle — and adds three things on top. Each one
replaces something the plain version left deliberately crude, and together they turn "select features
by trial and error" into "select features guided, at every step, by what the evidence actually says."
Those three additions are: a **tree-structured state**, a **personalized per-agent reward**, and
**interactive trainers** that coach the agents. The rest of this document is those three, plus the
family of variants used to isolate what each one contributes.

## 1. A tree-structured state

In simple RL the agents work from a minimal picture of the situation. Full IRFS replaces that with a
state built from a **graph of the currently selected features**. Each selected feature is a node
carrying a short descriptive vector of summary statistics computed on the training data
(`[mean, std, min, max]`). The edges encode structure between features in two ways: **signed
correlations** between features, and **directed edges taken from the Decision Tree** — so the tree's
own view of how features relate is literally wired into the state.

That graph is then run through a **graph convolution** (the reference method's Section 3.2, Steps
4–5). Each node's vector is updated from a mix of its tree-neighbors and the graph as a whole,
governed by a neighbor-vs-global blend factor (λ). In the running engine, the result is exposed as a
fixed-width, **feature-specific state row for each agent**. That preserves the feature's identity while
still letting its row reflect the currently selected graph. The state width stays constant even as the
subset size changes.

There are two encoders behind this seam:

- a **fixed-weight aggregator** — the current default — which combines node vectors using the signed
  correlations directly, with no learned parameters, and
- an optional **trainable GCN**, selected with `--state-encoder trained_gcn`, where the aggregation
  transform `W` is learned jointly with the agents' value estimates
  (`H' = activation(Â · H · W)`).

The trainable GCN is the central fidelity gain of the feature: it gives the state representation the
*capacity to adapt* to the data rather than folding features together by a fixed rule.

## 2. A personalized per-agent reward

Simple RL hands every agent the same single reward. Full IRFS starts from that same overall signal —
**`Acc − βR`**, the subset's validation accuracy minus a redundancy penalty `R` (weighted by β) — but
then **splits it across the agents** so each learns from its own tailored signal (Section 3.3).

The headline scheme weights each *selected* agent by its feature's Decision-Tree importance:
**`rᵢ = Iᵢ · (Acc − βR)`**. A feature the tree relied on heavily earns a larger share of the reward; a
feature that was not selected this step earns **exactly zero**, regardless of its history. (An
alternate scheme weights by each feature's historical selection frequency instead; the study's default
is the importance scheme, which needs no history.) The effect is that credit for a good subset flows
preferentially to the features that actually carried it, rather than being spread evenly across every
agent that happened to be in the subset.

## 3. Interactive trainers — the Decision Tree in the loop

This is the piece that gives the method its name. In simple RL, once the agents vote, the votes stand.
In full IRFS, each step has an extra beat: after every agent has voted, the run identifies the
**hesitant agents** — features that participated in the previous subset but whose agents now vote to
deselect them — and lets an active **trainer** override just those hesitant votes, nudging them toward
what the evidence favors. Stable keeps and newly selected features are left to the agents' own votes.

The advice reaches the engine through a pluggable **action-advisor seam**: the engine decides *where*
advice lands (it applies a `{feature: action}` map to the votes before taking the union of "keep"
votes), while the advisor decides *what* the advice is. Three trainers can sit behind that seam:

- a **relevance trainer**, coaching hesitant agents by each feature's relevance to the label,
- a **Decision-Tree-importance trainer**, coaching by the probe tree's importances, and
- a **hybrid scheduler**, which sequences the two trainers across the run's steps rather than
  committing to one.

For the IRFS variants, coaching is entirely a matter of which advisor (if any) is plugged in. The
separate `marlfs` baseline deliberately does **not** share that richer state/reward configuration: it
keeps the minimal state, uniform overall reward, and no advisor.

## The five reinforced variants

The code registers five reinforced method names. They are not five identical IRFS configurations with
different advisors:

| Variant | State and reward | Coaching applied |
| --- | --- | --- |
| **`marlfs`** | Minimal state and uniform overall reward | none — this is [simple RL](03-simple-rl.md) |
| **`no_trainer`** | Configured tree/graph state and personalized reward | none |
| **`relevance`** | Same IRFS state and reward | relevance trainer |
| **`dt_importance`** | Same IRFS state and reward | Decision-Tree-importance trainer |
| **`full_irfs`** | Same IRFS state and reward | hybrid teaching: relevance, then DT importance, then withdrawal |

`full_irfs` is the hybrid configuration; there is no separate executable `hybrid` method because it
would be identical. The default run includes `marlfs`, `no_trainer`, and `full_irfs`; `relevance` and
`dt_importance` are diagnostic ablations enabled with `--diagnostic-ablations`.

## Fair by construction

Everything above stays inside the same guarantees as every other method in the study. All of the
signals are computed on non-test data only — the state and correlations from training, the reward's
accuracy from validation, the importances from the train-fit probe — and the held-out test partition
is never touched during the run. Every source of randomness draws from the one shared seed, so a given
run reproduces the same subsets and the same per-step trajectories exactly. Full IRFS is more
elaborate than the others, but it is measured on precisely the same footing — which is the only way
its result means anything.
