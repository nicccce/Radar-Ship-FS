"""Independent deletion-control checks and five sampled estimator refits."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from audit_search_diagnosis import manual_bacc
from run_search_diagnosis import ROOT, load_raw, read, read_lines, sha
from run_unified_baselines import _write_json


def main():
    prereg = read(ROOT / "ablation-preregistered.json")
    assert prereg["code_sha256"] == sha("src/audit_search_ablation.py")
    assert prereg["protocol_sha256"] == sha("documents/research-plan/search-diagnosis-ablation-addendum.md")
    controls = read_lines(ROOT / "ablation/controls.jsonl")
    assert len(controls) <= 560
    assert sum(c["search_fit_count"] for c in controls) == 5 * len(controls)
    assert sum(c["validation_fit_count"] for c in controls) == 2 * len(controls)
    pairs = pd.read_csv(ROOT / "ablation/low_mi_contributions.csv")
    raw, labels = load_raw()
    errors = []
    for seed in (42, 43, 44, 45, 46):
        case = ROOT / "search" / f"nested_dev-seed-{seed}"
        context, cs = read(case / "context.json"), read_lines(case / "candidates.jsonl")
        val = {
            r["candidate_id"]: r for r in read_lines(ROOT / "validation" / f"seed-{seed}" / "scores.jsonl")
        }
        lookup = {tuple(c["clean_indices_0based"]): c for c in controls if c["seed"] == seed}
        for key, c in lookup.items():
            assert [context["final_feature_ids"][i] for i in key] == c["original_feature_ids_1based"]
            assert abs(np.mean(c["fold_scores"]) - c["search_accuracy"]) < 1e-12
        for r in pairs[pairs.seed == seed].itertuples():
            s, t = cs[r.anchor_id], cs[r.candidate_id]
            ctrl = lookup[tuple(sorted(set(s["clean_indices_0based"]) & set(t["clean_indices_0based"])))]
            errors.extend(
                [
                    abs(r.conditional_search_gain - (t["search_accuracy"] - ctrl["search_accuracy"])),
                    abs(r.conditional_lr_gain - (val[r.candidate_id]["lr_bacc"] - ctrl["lr_bacc"])),
                ]
            )
        # One predetermined control per split, rather than the best-looking control.
        c = lookup[sorted(lookup)[0]]
        cols = np.array(c["original_feature_ids_1based"]) - 1
        scores = []
        for fold in context["fold_original_rows"]:
            tr, va = fold["fit"], fold["held_out"]
            clf = DecisionTreeClassifier(random_state=context["cv_tree_random_state"])
            clf.fit(raw[tr][:, cols], labels[tr])
            scores.append(np.mean(clf.predict(raw[va][:, cols]) == labels[va]))
        errors.append(abs(np.mean(scores) - c["search_accuracy"]))
        tr, va = context["development_original_rows"], context["validation_original_rows"]
        scaler = StandardScaler().fit(raw[tr][:, cols])
        clf = LogisticRegression(
            C=1, solver="liblinear", max_iter=5000, class_weight="balanced", random_state=seed
        )
        clf.fit(scaler.transform(raw[tr][:, cols]), labels[tr])
        pred = clf.predict(scaler.transform(raw[va][:, cols]))
        errors.append(abs(manual_bacc(labels[va], pred) - c["lr_bacc"]))
        dt = DecisionTreeClassifier(random_state=seed).fit(raw[tr][:, cols], labels[tr])
        pred = dt.predict(raw[va][:, cols])
        errors.append(abs(np.mean(pred == labels[va]) - c["dt_accuracy"]))
    assert max(errors) < 1e-12
    _write_json(
        ROOT / "ablation/audit.json",
        {
            "passed": True,
            "controls_checked": len(controls),
            "contribution_rows_checked": len(pairs),
            "sampled_refitted_controls": 5,
            "refit_count": 35,
            "max_error": max(errors),
            "preregistration_hashes_verified": True,
        },
    )
    print(read(ROOT / "ablation/audit.json"))


if __name__ == "__main__":
    main()
