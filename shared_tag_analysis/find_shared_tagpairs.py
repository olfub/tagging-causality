#!/usr/bin/env python3
"""Find tag pairs (independent of direction) that are shared across datasets."""
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "tagpairs.csv"
OUTPUT_FILE = Path(__file__).parent / "shared_tagpairs.csv"


def main():
    with INPUT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        entries = [
            {
                "dataset": row["dataset"],
                "tagA": row["tagA"],
                "tagB": row["tagB"],
                "support": int(row["support"]),
            }
            for row in reader
        ]

    groups = defaultdict(list)
    for entry in entries:
        key = frozenset((entry["tagA"], entry["tagB"]))
        groups[key].append(entry)

    results = []
    for key, group_entries in groups.items():
        if len(key) != 2:
            continue  # skip self-pairs, if any
        for e1, e2 in combinations(group_entries, 2):
            if e1["dataset"] == e2["dataset"]:
                continue

            # pick the entry with higher support as the "stronger" one (datasetA)
            if e1["support"] > e2["support"]:
                strong, weak = e1, e2
            elif e2["support"] > e1["support"]:
                strong, weak = e2, e1
            else:
                # tie-break deterministically by dataset name
                strong, weak = sorted((e1, e2), key=lambda e: e["dataset"])

            agree = (strong["tagA"] == weak["tagA"]) and (strong["tagB"] == weak["tagB"])

            results.append(
                (strong["dataset"], weak["dataset"], strong["tagA"], strong["tagB"], agree)
            )

    results.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datasetA", "datasetB", "tagA", "tagB", "agree"])
        writer.writerows(results)

    print(f"Wrote {len(results)} shared tag-pair rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
