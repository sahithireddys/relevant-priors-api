"""
Local evaluator for the downloaded public JSON.

Usage:
  python eval_local.py path/to/relevant_priors_public.json
"""

import json
import sys

from app.prior_relevance import predict_prior_relevance


def build_truth(data):
    if "truth" in data:
        return {
            (str(row["case_id"]), str(row["study_id"])): bool(row["is_relevant_to_current"])
            for row in data["truth"]
        }

    # Fallback for alternate JSON layouts where labels live inside prior_studies.
    truth = {}
    for case in data.get("cases", []):
        case_id = str(case["case_id"])
        for prior in case.get("prior_studies", []):
            for key in ("is_relevant_to_current", "is_relevant", "relevant", "label"):
                if key in prior:
                    truth[(case_id, str(prior["study_id"]))] = bool(prior[key])
                    break
    return truth


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    truth = build_truth(data)
    total = 0
    correct = 0
    false_pos = 0
    false_neg = 0
    examples = []

    for case in data.get("cases", []):
        case_id = str(case["case_id"])
        current = case["current_study"]
        for prior in case.get("prior_studies", []):
            key = (case_id, str(prior["study_id"]))
            if key not in truth:
                continue
            y_true = truth[key]
            y_pred = predict_prior_relevance(current, prior)
            total += 1
            correct += int(y_true == y_pred)
            false_pos += int(y_pred is True and y_true is False)
            false_neg += int(y_pred is False and y_true is True)
            if y_true != y_pred and len(examples) < 20:
                examples.append({
                    "case_id": case_id,
                    "current": current.get("study_description"),
                    "prior": prior.get("study_description"),
                    "true": y_true,
                    "pred": y_pred,
                })

    print(f"cases={len(data.get('cases', []))}")
    print(f"truth={len(truth)}")
    print(f"evaluated={total}")
    print(f"correct={correct}")
    print(f"accuracy={correct / total:.4f}" if total else "accuracy=n/a")
    print(f"false_positive={false_pos}")
    print(f"false_negative={false_neg}")
    print("\nFirst errors:")
    for e in examples:
        print(json.dumps(e, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_local.py path/to/relevant_priors_public.json")
        sys.exit(1)
    main(sys.argv[1])
