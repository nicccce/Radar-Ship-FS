# Project Overview

## The goal

When you build a model on tabular data, one of the first questions is: *which columns actually
matter?* Picking the right subset of features can make a model simpler, faster, and more accurate —
and there are many competing ways to make that choice.

A research paper made an intriguing claim: if you put a **Decision Tree "in the loop"** of a
reinforcement-learning feature selector — letting the tree *coach* the learner about which features
look important — you end up with better feature subsets than either the classical selection methods
or plain reinforcement learning that gets no coaching. It's a compelling idea, but it had only ever
been tried on the paper's own datasets, and there was no reusable implementation to check it with.

**This project exists to settle that claim.** It builds one honest, dataset-agnostic pipeline that
puts every method to the same test and produces a single comparison table you can trust. The emphasis
is on *fairness*: every method sees the same data split, is scored by the same yardstick, and is kept
away from the same walled-off test data. If the methods weren't measured identically, the comparison
would answer nothing — so equal footing is the whole point.

The plan is to **reproduce first, then generalize**: confirm the result on a well-known dataset, and
because the pipeline is driven entirely by configuration, later re-run the exact same comparison on
harder data just by pointing it at a different dataset.

## The pipeline at a glance

Everything the project does is expressed as a single run. You give it two things — **which dataset**
to use and **which random seed** to run under — and it does the rest, reading top to bottom like a
story: prepare the data fairly, run every method on that same footing, score them all the same way,
and write out one comparison you can reproduce exactly.

That single run is deliberately a *thin narrator*: it doesn't invent any of the selection logic
itself, it simply wires together the real pieces and lets each method do its work through one shared
"produce a subset" contract. Because every method flows through the same path, the resulting numbers
are directly comparable — which is the entire reason the pipeline is shaped this way.

A run unfolds in three phases: **Setup** prepares the data and the scoring yardstick, the **Full
comparison** puts all the methods head-to-head and records the verdict, and the **Cross-seed
aggregate** repeats the whole thing across several seeds so the answer reflects a stable pattern
rather than one lucky (or unlucky) split. The sections below walk through each in turn.

## The phases of the flow

### 1. Setup

Before any method runs, the pipeline lays the fair, repeatable ground everything else stands on. It
reads the configuration and fixes the random seed, so the run can be reproduced exactly later. It
loads the chosen dataset and splits it into parts — the data the methods are allowed to learn from
and choose features on, and a separate slice of **test data that is locked away** and never touched
during selection. That quarantine is what keeps any result from being secretly contaminated by the
very data it will later be judged on.

Setup also builds the single **scoring yardstick**: a Decision Tree that takes any chosen subset of
features and turns it into an accuracy number. Every method, later, is measured with this same probe
on this same split. By the end of Setup there is one shared, leakage-safe context — and the harder
"is this even apples-to-apples?" question has already been answered, once, for everyone.

### 2. Full comparison & held-out validation

This is the heart of the run: every method competes on the exact same footing in one **canonical
comparison**. Two families line up side by side. On one side are the **four classical methods** —
established, well-understood ways of ranking or pruning features. On the other are the three
**headline reinforced methods**: `marlfs` (plain RL), `no_trainer` (IRFS state and reward without
advice), and `full_irfs` (the hybrid-teaching headline). The two single-trainer variants,
`relevance` and `dt_importance`, are diagnostic ablations and run only with
`--diagnostic-ablations`.

So the default canonical comparison contains **seven methods**; a diagnostic run contains **nine**.
All included methods share the same split, validation surface, and Decision-Tree probe. Only once every
method has made its final choice is the **locked-away test data released — exactly once** — to give each
chosen subset an honest final grade. The phase then prints the headline table and writes the
**reproducible artifact**: configuration, dataset identity, selected features, per-step metrics, final
comparison, and fidelity notes.

Before that canonical comparison, the command-line program also prints a classical-baseline and
interactive-feedback walkthrough. Those preliminary selections make the progress narrative visible;
the full-comparison pass is the one whose results are persisted and aggregated.

### 3. Cross-seed aggregate

A single run rests on one random split of the data, and any one split can be a little lucky or
unlucky. So the pipeline repeats the whole comparison across several seeds, then gathers the results
into one combined view. Instead of reading a single number for each method, you see how it performs
*across* runs — and, most importantly, how the headline "full IRFS" method stacks up against the
classical field on average rather than on one roll of the dice.

This is what turns the study's answer from an anecdote into something you can stand behind: the
verdict reflects a stable pattern, not a single noisy split.

## What you get out of it

Every run leaves behind a **reproducible artifact** — a durable record of exactly what happened: the
configuration and seed, the dataset it ran on, the features each method chose, how each scored along
the way and on the held-out test data, the final head-to-head comparison, and the fidelity notes that
own up to every judgment call made where the paper was silent. Anyone can re-run it and get the same
numbers.

Taken together, that record answers the question the project set out to settle: **does
Decision-Tree-coached reinforcement learning really pick better features than the classical methods
and plain RL?** — first on the reference dataset, and then, by changing a single configuration
setting, on whatever harder dataset comes next. The pipeline doesn't just produce an answer; it
produces one you can trust and reuse.
