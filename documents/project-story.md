# Project Story

A research paper claims that putting a Decision Tree "in the loop" of a reinforcement-learning feature selector can produce better feature subsets than both classical selection methods and plain, non-interactive RL. The Decision Tree acts as a coach: it helps the learner understand which features matter, how selected features behave structurally, and how useful a subset appears during search.

The catch is that the claim needs a reusable, fair, dataset-agnostic implementation to test it outside the paper's original setting. This project exists to build that implementation and run the comparison carefully: first on WDBC, then on harder datasets by configuration alone.

The intended result is concrete: one run, every method scored the same way, and one comparison table a reader can trust.

## What We Are Building

The project builds a configurable feature-selection pipeline for tabular classification datasets. It compares nine methods on equal footing:

- **Four classical baselines:** relevance-ranked top-k, recursive Decision-Tree-importance elimination, mRMR, and L1 / LASSO.
- **Five reinforced methods:** full IRFS plus reinforced variants that remove or change trainer guidance, including no-coach, relevance-only, tree-importance-only, and hybrid guidance.

Every method is routed through the same leakage-safe data split and scored with the same downstream Decision Tree. The run emits one reproducible artifact containing the effective configuration, dataset name/source, selected subsets, per-step reinforced metrics, final Best and Average Accuracy comparison, and fidelity notes for implementation choices that fill gaps in or depart from the reference method.

The point of this discipline is simple: if methods are not measured identically, the comparison does not answer the research question.

## Why The Work Is Ordered This Way

The strategy is vertical-slice-first. The project starts with the measurement substrate before building the full IRFS machinery.

That order matters because the central concern is not only whether the reinforced method is clever. The deeper concern is whether very different method families can compose through one shared "produce a subset" contract, whether the scoring path is genuinely fair, whether test data stays quarantined from search and learning, and whether the RL engine converges on small data instead of collapsing to trivial behavior.

So the work is ordered to strengthen the foundation first:

1. **Build the measurement spine:** configuration, seed control, dataset loading, leakage-safe splitting, Decision Tree scoring, and windowed accuracy metrics.
2. **Prove the harness with a thin comparison:** run one classical method and plain no-coach RL end to end through one orchestrator and one artifact.
3. **Complete the classical baselines:** add the remaining classical selectors and pin the mRMR dependency.
4. **Add the IRFS feedback path:** introduce trainers, tree-structured state, and personalized reward on top of the already-proven engine.
5. **Run the full comparison on WDBC:** execute all nine methods and produce the headline comparison.

This means the most interesting IRFS result arrives late. The tradeoff is intentional: by the time the expensive feedback machinery is built, the measurement ground under it is already solid.

```text
IRFS Reproduction & Generalization Study
├── Phase 1  Measurement Spine ........... fair, reproducible scoring substrate
├── Phase 2  Equal-Footing Harness ....... thin classical-vs-RL comparison
├── Phase 3  Classical Baseline Suite .... the remaining classical methods
├── Phase 4  IRFS Feedback ............... trainers, tree-state, personalized reward
└── Phase 5  Full Comparison + WDBC ...... all nine methods on dataset #1
```

## Phase Journey

### Phase 1: Measurement Spine

This phase establishes the substrate that lets any feature subset be scored fairly and reproducibly. It walls off test data from feature search and learning, introduces repeatable seed behavior, and provides a shared Decision Tree probe that turns a subset into accuracy, feature importances, and tree structure.

Its payoff is to settle the most basic question for every later phase: are methods being compared apples-to-apples?

### Phase 2: Equal-Footing Harness

This phase proves that a classical selector and an RL selector can both run through the same subset contract and scoring path. It also gives the first early signal about whether the RL loop converges on small data.

The minimal state encoding and reward used here are deliberately provisional. They provide an integration spine that later phases can replace without disturbing the overall engine.

### Phase 3: Classical Baseline Suite

This phase completes the classical side of the comparison. It adds the remaining classical selectors and pins library behavior where external implementations could otherwise drift between machines.

Its payoff is a reproducible field of classical contenders.

### Phase 4: IRFS Feedback

This is the heart of the study. It adds the three capabilities that make full IRFS distinct from plain RL:

- trainers that coach hesitant agents toward stronger features
- a tree-structured state representation based on feature correlation and Decision Tree edges
- a personalized per-agent reward

After this phase, the central hypothesis becomes testable rather than assumed.

### Phase 5: Full Comparison + WDBC Validation

This phase runs every method together on the same shared split, computes the Best and Average Accuracy comparison, writes fidelity notes, and validates the full pipeline on WDBC.

Its payoff is the first complete answer: whether Decision-Tree-coached RL selects better feature subsets than the classical and plain-RL alternatives on the first dataset.

## The Whole Story

The five phases follow one line of reasoning:

```text
Can we measure fairly?
-> Do the method families compose on that fair footing?
-> Is the classical side complete and reproducible?
-> Does interactive Decision Tree feedback help?
-> What is the verdict on WDBC, and is the pipeline ready to generalize?
```

The measurement spine makes later numbers trustworthy. The harness confirms the comparison is fair before expensive machinery is added. Classical breadth and IRFS depth bring in the full field of contenders. The final run turns the project into the thing it exists to produce: a reproducible, honest comparison of whether Decision-Tree-coached RL really selects better features than the classical and plain-RL alternatives.
