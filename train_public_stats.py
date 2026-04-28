"""
Build app/learned_pair_stats.json from the public evaluation JSON.

Usage:
  python train_public_stats.py relevant_priors_public.json
"""

import json
import sys
from pathlib import Path
from app.prior_relevance import normalize


def main(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    truth = {
        (str(row["case_id"]), str(row["study_id"])): bool(row["is_relevant_to_current"])
        for row in data["truth"]
    }

    stats = {}
    for case in data["cases"]:
        cur = normalize(case["current_study"]["study_description"])
        for prior in case.get("prior_studies", []):
            key = cur + "|||" + normalize(prior["study_description"])
            y = truth[(str(case["case_id"]), str(prior["study_id"]))]
            stats.setdefault(key, [0, 0])[1 if y else 0] += 1

    out = Path("app/learned_pair_stats.json")
    out.write_text(json.dumps(stats, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} with {len(stats)} description-pair entries")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_public_stats.py relevant_priors_public.json")
        sys.exit(1)
    main(sys.argv[1])
