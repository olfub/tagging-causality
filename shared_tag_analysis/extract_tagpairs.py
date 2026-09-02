#!/usr/bin/env python3
"""Extract tag pairs from tagpairs.txt."""
import csv
import re
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "tagpairs.txt"
OUTPUT_CSV = Path(__file__).parent / "tagpairs.csv"

HEADER_RE = re.compile(r"\\textbf\{([^}]+)\}")
ROW_RE = re.compile(
    r"\\text\{(.+?)\}\s*&~\\rightarrow~\s*\\text\{(.+?)\}\s*&&\s*\((\d+)\\%\s*~/~\s*(\d+)~~\)"
)


def clean_tag(raw: str) -> str:
    # strip LaTeX quote markers (``tag'') and unescape underscores
    tag = raw.strip().strip("`").strip("'")
    return tag.replace("\\_", "_")


def parse(lines):
    dataset = None
    rows = []

    for line in lines:
        header_match = HEADER_RE.search(line)
        if header_match:
            dataset = header_match.group(1).strip()
            continue

        row_match = ROW_RE.search(line)
        if row_match and dataset is not None:
            tag_a = clean_tag(row_match.group(1))
            tag_b = clean_tag(row_match.group(2))
            consistency = int(row_match.group(3))
            support = int(row_match.group(4))
            rows.append((dataset, tag_a, tag_b, consistency, support))

    return rows


def main():
    lines = INPUT_FILE.read_text().splitlines()
    rows = parse(lines)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "tagA", "tagB", "consistency", "support"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
