# Interactive RL Trainers

This document explains the **interactive-trainer layer** first introduced in Phase 2: advice applied
after agents vote and before their subset is committed. It intentionally isolates that one boundary.
In the current runtime, the advised IRFS variants — `relevance`, `dt_importance`, and `full_irfs` —
compose this layer with the richer tree/graph state and personalized reward described in [the
feedback-loop review](06-decision-tree-feedback-loop.md). The `no_trainer` IRFS variant deliberately
has no advisor, while the separate `marlfs` baseline remains the plain-RL configuration.

Simple RL already has the main reinforcement-learning loop: one agent per feature, `SELECT` /
`DESELECT` votes, a subset formed from those votes, validation scoring with the Decision-Tree probe,
and temporal-difference updates that slowly improve the agents' value estimates. What it does not
have is outside guidance. Agents learn only by trial and error, so they can spend many steps exploring
weak subsets or dropping useful features too early.

Phase 2 adds one narrow intervention point:

```text
agents vote -> trainer may advise hesitant agents -> final votes form the subset
```

The trainer does not replace RL. The agents still vote first, the subset is still scored the same way,
and the agents still learn from the resulting transition. The trainer only gets a chance to revise
specific votes before the next subset is committed.

## The Core Idea

In simple RL, the agents vote and the votes immediately become the next subset. If a feature's agent
votes `DESELECT`, that feature is removed.

Phase 2 adds a second opinion before removing features that were already selected:

```text
"RL wants to drop this old feature.
Does a trainer still think it looks useful?"
```

If the trainer says yes, the vote can be changed back to `SELECT` for this step. So the simplest
purpose of Phase 2 is:

> Help RL avoid dropping useful features too early.

The trainer is not choosing the full subset. It only watches for one situation:

```text
previous step: feature was selected
current vote:  RL wants to deselect it
```

That feature is called **hesitant**. A new feature moving from `DESELECT` to `SELECT` is not trainer
work; that is just normal RL exploration.

So the trainer is an **action-advice** improvement. The advisor itself changes selected actions before
the next subset is committed; it does not implement a state representation or reward formula. Those
are supplied by the engine configuration around it.

## Participated, Assertive, and Hesitant Agents

The trainer mechanism depends on three sets of features:

- **Participated features** are the features selected in the previous step.
- **Assertive features** are participated features whose agents still vote `SELECT`.
- **Hesitant features** are participated features whose agents now vote `DESELECT`.

Only hesitant agents can receive advice.

The four possible action transitions are:

| Previous action | Current RL vote | Trainer role |
| --- | --- | --- |
| `SELECT` | `SELECT` | assertive |
| `SELECT` | `DESELECT` | hesitant |
| `DESELECT` | `SELECT` | normal new RL selection |
| `DESELECT` | `DESELECT` | not participating |

This distinction matters. A new `DESELECT -> SELECT` vote is not a hesitant feature in the paper. It
is simply RL trying to add a feature. The trainer should not treat every changed action as uncertain;
it should focus on features that were already in the subset and are now being removed.

## Where Advice Enters the Loop

Simple RL:

```text
encode state
agents vote
form subset from SELECT votes
score subset
assign reward
learn
advance
```

Phase 2:

```text
encode state
agents vote
classify previous/current actions
trainer advises hesitant agents
apply advice to the votes
form subset from final SELECT votes
score subset
assign reward
learn
advance
```

The advice is applied before the selected-feature union is built. That means an advised action affects
both the subset that gets scored and the action stored in the agent's experience memory.

## Trainer Types

Phase 2 has three trainer modes. They all do the same kind of job: look at a hesitant feature and
decide whether RL is about to drop it too soon.

### Relevance Trainer

The relevance trainer asks the simplest question:

> Does this feature, by itself, seem related to the label?

It uses a relevance score from the training data. If RL wants to drop a feature, but that feature still
looks more relevant than the features RL is confidently keeping, the trainer says:

```text
keep it for now
```

How this helps:

The relevance trainer protects obviously useful features from being removed too early. It is a fast,
simple teacher: it does not understand feature combinations deeply, but it can notice when RL is about
to throw away a feature that has a strong direct signal.

In short:

```text
Do not drop a feature that still looks strongly related to the target.
```

### Decision-Tree-Importance Trainer

The Decision-Tree-importance trainer asks a slightly richer question:

> Did the Decision Tree actually use this feature to make predictions?

It trains the probe tree on the previously selected features, then looks at feature importance. If RL
wants to drop a feature, but the tree relied on that feature more than it relied on the features RL is
confidently keeping, the trainer says:

```text
keep it for now
```

How this helps:

The Decision-Tree trainer protects features that matter inside a real predictive model. This is more
task-aware than simple relevance: a feature may not look best alone, but the tree may still find it
useful when combined with other selected features.

In short:

```text
Do not drop a feature that the Decision Tree is still using.
```

At this advisor boundary, the Decision Tree is used to decide whether to protect a hesitant feature.
In the current `dt_importance` and `full_irfs` configurations, that advice is also composed with the
richer state and personalized reward described in the next review.

### Hybrid Trainer

The hybrid trainer asks:

> Can we use the simple teacher first, then the model-aware teacher, then let RL continue alone?

It does not introduce a third kind of advice. It schedules the two trainers:

```text
early steps      -> relevance trainer
middle steps     -> Decision-Tree-importance trainer
later steps      -> no trainer
```

How this helps:

Early in training, RL is still noisy, so the fast relevance trainer gives broad guidance. Later, the
Decision Tree trainer gives more task-specific guidance. After that, guidance is withdrawn so the RL
agents do not depend on the trainers forever.

In short:

```text
Start with simple guidance, switch to smarter guidance, then remove guidance so RL continues by itself.
```

## What Changes and What Does Not

The trainer layer changes:

- a trainer/advisor seam is added after voting;
- agents are classified as participated, assertive, or hesitant;
- trainers can override hesitant `DESELECT` votes back to `SELECT`;
- advisor configurations can be compared cleanly: no trainer, relevance, Decision-Tree importance,
  and Hybrid Teaching (`full_irfs`).

The trainer layer itself does not change:

- the basic per-feature agent setup;
- the `SELECT` / `DESELECT` action space;
- the subset scoring path;
- the minimal state representation;
- the reward formula;
- the final return contract.

That is the clean boundary: **the trainer changes the action before the subset is committed; the
feedback layer changes what the agents observe and how they are rewarded.** In the current code those
layers are composed for `relevance`, `dt_importance`, and `full_irfs`; `full_irfs` is the Hybrid
Teaching schedule, so there is no additional `hybrid` registry name.

## Why This Helps

Simple RL can waste time because early value estimates are noisy. An agent may drop a useful feature
before it has enough experience to understand its value. The trainer gives the run a conservative
correction: if a previously selected feature still looks strong by an external criterion, keep it for
now.

This does not guarantee the feature stays forever. The same feature can be dropped later if the agent
and trainer no longer favor it, or if trainer guidance has been withdrawn. The point is not to freeze
the subset; it is to reduce avoidable poor moves while the agents are still learning.

## Fidelity Note

For paper fidelity, "hesitant" must be interpreted narrowly:

```text
SELECT -> DESELECT
```

It should not mean "any changed action." Treating `DESELECT -> SELECT` as hesitant would make trainers
coach newly added features too, which is broader than the paper's trainer-agent interaction. In the
faithful Phase 2 version, new selections remain ordinary RL exploration, while trainers focus only on
previously selected features that are about to be removed.

## Slide Content

**Title:** Phase 2 — Trainers Give RL a Second Opinion

**Core idea:**

RL still votes first. But before a previously selected feature is removed, a trainer gives a second
opinion:

```text
SELECT before -> RL now says DESELECT -> trainer may say "keep it for now"
```

So Phase 2 helps RL avoid dropping useful features too early, without letting the trainer choose the
whole subset.

**Two trainer signals:**

```text
1. Relevance Trainer
   Looks at the feature itself:
   "Is this feature strongly related to the label?"

2. Decision-Tree Trainer
   Looks at the downstream model:
   "Did the tree actually use this feature?"
```

**Hybrid teaching:**

```text
Early training:  use Relevance Trainer
Middle training: use Decision-Tree Trainer
Late training:   remove guidance so RL continues by itself
```

**Takeaway:**

The trainers do not choose the whole subset. They only help RL avoid dropping useful features too
early.
