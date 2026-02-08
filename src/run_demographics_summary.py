import argparse
import json
import os
from datetime import datetime

import numpy as np


def load_patients(json_dir):
    patients = []
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("patient", []):
            patients.append(p)
    return patients


def compute_age(birthdate, ref_date):
    if not birthdate:
        return None
    try:
        b = datetime.strptime(birthdate, "%Y-%m-%d")
    except ValueError:
        return None
    age = ref_date.year - b.year - ((ref_date.month, ref_date.day) < (b.month, b.day))
    return age


def main():
    parser = argparse.ArgumentParser(description="Demographic summary from JSON files.")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    patients = load_patients(args.json_dir)
    ref_date = datetime.today()

    genders = []
    ages = []
    hy_stages = []
    item10_scores = []
    for p in patients:
        g = p.get("gender")
        if g:
            genders.append(g)
        age = compute_age(p.get("birthdate"), ref_date)
        if age is not None:
            ages.append(age)
        if "hoehn_yahr_stage" in p:
            hy_stages.append(p["hoehn_yahr_stage"])
        items = p.get("mds_updrs_part3", {}).get("itmes", [])
        if items:
            v = items[0].get("10")
            if v is not None:
                item10_scores.append(int(sum(v) if isinstance(v, list) else v))

    gender_counts = {}
    for g in genders:
        gender_counts[g] = gender_counts.get(g, 0) + 1

    age_stats = {}
    if ages:
        age_stats = {
            "n": int(len(ages)),
            "mean": float(np.mean(ages)),
            "std": float(np.std(ages, ddof=1)) if len(ages) > 1 else 0.0,
            "min": int(np.min(ages)),
            "max": int(np.max(ages)),
        }

    summary = {
        "total_patients": int(len(patients)),
        "gender_counts": gender_counts,
        "age_stats": age_stats,
        "age_histogram": {},
        "hy_stage_counts": {},
        "item10_counts": {},
    }

    if ages:
        bins = [0, 50, 60, 70, 80, 90, 200]
        hist, edges = np.histogram(ages, bins=bins)
        summary["age_histogram"] = {
            f"{edges[i]}-{edges[i+1]}": int(hist[i]) for i in range(len(hist))
        }

    if hy_stages:
        hy_counts = {}
        for v in hy_stages:
            key = str(v)
            hy_counts[key] = hy_counts.get(key, 0) + 1
        summary["hy_stage_counts"] = hy_counts

    if item10_scores:
        item10_counts = {}
        for v in item10_scores:
            key = str(v)
            item10_counts[key] = item10_counts.get(key, 0) + 1
        summary["item10_counts"] = item10_counts

    out_path = args.out_path or os.path.join(args.json_dir, "demographics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved demographics summary to {out_path}")


if __name__ == "__main__":
    main()
