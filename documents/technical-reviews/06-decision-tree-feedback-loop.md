# Decision Tree Feedback Loop

This document explains the Decision-Tree **feedback layer** first introduced in Phase 3. It is
separate from, but composable with, the trainer layer: trainers advise actions before a subset is
committed, while tree feedback supplies the state and reward information used around each step. Both
layers are active together in the advised IRFS configurations — `relevance`, `dt_importance`, and
`full_irfs`. The `no_trainer` IRFS variant uses the feedback layer without advice, while `marlfs`
keeps the minimal state and uniform reward.

The short version:

```text
Phase 2: a trainer helps before the action is committed.
Phase 3: the Decision Tree gives feedback after the subset is evaluated.
```

## The Core Idea

Simple RL mostly learns from the final score of a subset:

```text
Was this selected feature subset good or bad?
```

But when the Decision Tree evaluates that subset, it produces more than an accuracy number. It also
reveals which selected features it used, which ones mattered more, and how features appear together
inside the tree.

Phase 3 asks:

> If the Decision Tree already discovered useful structure about the selected features, why throw that
> structure away?

So the Decision Tree becomes a feedback source. RL still selects feature subsets, but after each
subset is evaluated, the tree's internal knowledge is fed back into the learning process.

In plain terms:

```text
RL proposes a subset.
The Decision Tree evaluates it.
The tree explains which features mattered.
RL uses that explanation to learn better.
```

## What Changes From Phase 2

Phase 2 changes the action before the next subset is committed:

```text
RL wants to drop a previous feature -> trainer may say "keep it for now"
```

Phase 3 changes the learning signal after the subset is evaluated:

```text
Decision Tree evaluated the subset -> use what the tree learned to improve state and reward
```

That boundary is important. Trainers are about **advice before committing an action**. Decision Tree
feedback is about **learning more from the result**.

## What Feedback Comes From The Decision Tree

After a subset is selected, the Decision Tree can provide three useful signals:

- **accuracy**: how well the subset performed;
- **feature importance**: which selected features the tree relied on most;
- **tree structure**: how selected features appear in the tree's split hierarchy.

Simple RL already uses the first signal, accuracy. Phase 3 adds the other two signals back into RL.

## Feedback Path 1: Better State Representation

In simple RL, each agent sees a small state built from basic feature-selection ideas such as relevance
and redundancy. That is useful, but it is a very small view of the situation.

Phase 3 builds a richer state from the selected subset.

The key shift is that the selected subset is no longer treated as a flat list of feature IDs. Phase 3
represents it as a graph: selected features are nodes, feature statistics are node attributes, and
correlations plus Decision Tree structure become edges. The engine sends each DQN a fixed-size,
**feature-specific row** derived from that graph.

The important distinction is:

```text
Nodes describe individual features.
Edges describe the selected-subset context around them.
```

So feature identity comes from the node, while feature context comes from the edges. This is what lets
an agent see not only "what my feature looks like," but also "how my feature relates to the other
features currently selected."

Both Phase 3 state modes use this same graph idea. The fixed tree/graph encoder applies a fixed
aggregation rule; the trainable GCN encoder learns the aggregation. The detailed mode comparison lives
in the [deep-dive note](decision-treen-feedback-loop-deep-dive.md).

The selected features are represented as a graph:

- each selected feature is a node;
- feature-feature correlations become graph connections;
- Decision Tree structure adds directed, task-aware connections;
- each feature node carries simple statistics from the training data.

Then the graph is converted into fixed-length state rows. Selected features exchange information
through correlation edges and tree edges; each selected agent receives its own updated node row.
Deselected agents receive a fixed-width correlation-based attachment to the selected graph, so every
agent still has a state of the same dimension.

The simple intuition:

```text
Do not show RL only "which features are selected."
Show RL how the selected features relate to each other,
and how the Decision Tree used them.
```

This helps because a feature's value may depend on context. A feature can be weak alone but useful
with another feature, or redundant when another selected feature already carries the same signal. The
tree/graph state gives RL a better picture of that context.

### Slide Content: Fixed Tree/Graph Encoder

**Title:** Better State Representation

**Core idea:**

Represent the selected subset as a graph, then use a fixed formula to turn that graph into the DQN
state.

**Visual flow:**

```text
Selected subset
      |
      v
Selected features become nodes
      |
      v
Node attributes: [mean, std, min, max]
      |
      v
Edges:
- correlations between features
- Decision Tree structure
      |
      v
Fixed graph aggregation
      |
      v
Per-agent DQN input: [g1, g2, g3, g4]
```

**What `[g1, g2, g3, g4]` means:**

```text
4 feature-specific graph values
derived from that feature's [mean, std, min, max]
after mixing with selected-feature context
```

**Node vs edge meaning:**

```text
Node = feature-oriented
Edge = context-oriented

Feature identity comes from nodes.
Feature context comes from edges.
```

**Key point:**

```text
No learned graph weights.
The encoder uses a fixed graph formula.
```

**Boundary note:**

```text
Selected features build the graph.
Each agent receives its own feature-specific state row.
```

## Feedback Path 2: Smarter Reward Assignment

Simple RL gives every agent the same reward for a subset. If the subset performs well, all agents get
the same signal; if it performs poorly, all agents get the same signal.

That is crude because not every selected feature contributed equally.

Phase 3 keeps the overall reward idea:

```text
overall_reward = accuracy - beta * redundancy
```

Then it splits that reward using Decision Tree feature importance:

```text
if feature i was selected:
    reward_i = importance_i * overall_reward

if feature i was not selected:
    reward_i = 0
```

Compactly:

```text
r_i = I_i * (Acc - beta * R),  if a_i = 1
r_i = 0,                       if a_i = 0
```

Simple meaning:

```text
important selected feature   -> larger share of reward
unimportant selected feature -> smaller share of reward
deselected feature           -> no participation credit
```

The simple intuition:

```text
Do not reward every selected feature equally.
Give more credit to the selected features the tree actually used.
```

The zero for a deselected feature does not mean "good job deselecting." It means the feature was not in
the evaluated subset, so it gets no credit for that round. Deselection is learned indirectly over many
rounds, by comparing zero against the rewards the agent gets when it does select.

This helps RL learn better credit assignment. Instead of telling every agent "the whole subset was
good," Phase 3 tells each selected agent roughly how much its own feature helped inside the downstream
model.

### Slide Content: Smarter Reward Assignment

**Title:** Smarter Reward Assignment

**Problem + core idea:**

Simple reward gives every selected feature the same credit. The improved reward gives more specific
credit by using Decision Tree importance.

```text
Strong selected feature -> more credit
Weak selected feature   -> less credit
Unselected feature      -> no participation credit
```

**Formula:**

```text
r_i = I_i * (Acc - beta * R),  if a_i = 1
r_i = 0,                       if a_i = 0
```

```text
I_i = tree importance
Acc = subset accuracy
R   = subset redundancy
a_i = selected/deselected action
```

**Zero reward note:**

```text
0 does not mean "good deselection."
0 means "this feature was not evaluated in this subset."
```

**Key point:**

```text
Do not give every selected feature equal credit.
Give more credit to the selected features the tree actually used.
```

## The Full Phase 3 Loop

The loop still starts with RL selecting a subset. The difference is what happens after the Decision
Tree evaluates that subset.

```text
RL agents select a feature subset
        |
        v
Decision Tree trains/evaluates that subset
        |
        v
extract tree feedback:
accuracy, feature importance, tree structure
        |
        v
build the next richer graph/tree-based state
        |
        v
assign personalized rewards using tree importance
        |
        v
RL updates and selects again
```

So Phase 3 is not just "use a Decision Tree as the scorer." Simple RL already scores subsets with a
Decision Tree probe. The new idea is to reuse the tree's internal knowledge as part of the RL loop.

## What Changes and What Does Not

Phase 3 changes:

- the IRFS state is tree/graph-informed instead of minimal (fixed-weight by default; trainable GCN is
  optional);
- Decision Tree structure can shape the state representation;
- Decision Tree feature importance can shape per-agent rewards;
- agents can learn from richer feedback than a single shared score.

Phase 3 does not change:

- the basic feature-agent setup;
- the `SELECT` / `DESELECT` action space;
- the idea that RL proposes subsets over many steps;
- the held-out test boundary;
- the Phase 2 trainer seam, which remains an action-advice layer.

## Why This Is The Paper's Main Feedback Loop

The paper's key observation is that the downstream model produces knowledge, not just a score.

If we only keep the accuracy, the Decision Tree says:

```text
this subset scored 0.93
```

If we keep the feedback, the Decision Tree also says:

```text
these features were important
these features were connected in the tree
these selected features deserve more credit
```

That is the heart of Phase 3:

```text
Learn not only from the outcome,
but also from the knowledge generated while producing the outcome.
```

## Slide Content

**Title:** Phase 3 - Decision Tree Feedback Loop

**Core idea:**

The Decision Tree does more than score a subset. It explains which selected features mattered, and RL
uses that explanation to learn better.

```text
RL selects subset -> Decision Tree evaluates -> tree feedback improves RL
```

**Two feedback paths:**

```text
1. Better state
   Use feature correlations + tree structure
   so RL sees how selected features relate.

2. Smarter reward
   Use tree importance
   so useful selected features get more credit.
```

**Boundary from Phase 2:**

```text
Phase 2: trainers advise actions before commit.
Phase 3: Decision Tree feedback improves learning after evaluation.
```

**Takeaway:**

The tree is no longer just a scorer. It becomes part of the learning loop.
