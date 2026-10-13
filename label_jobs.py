"""
Labelling workflow:

  1. Export jobs to CSV for labelling in Excel / Google Sheets:
       uv run python label_jobs.py --export

  2. Open jobs_for_labelling.csv, fill the 'relevance' column:
       0 = not relevant   1 = somewhat relevant   2 = highly relevant
     Leave blank for any job you want to skip.

  3. Import labels back:
       uv run python label_jobs.py --import

  Re-running --export preserves existing labels in the 'relevance' column.
"""

import argparse
import csv
import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "jobs_snapshot.json"
LABELS_PATH = Path(__file__).parent / "labels.json"
CSV_PATH = Path(__file__).parent / "jobs_for_labelling.csv"

CSV_COLUMNS = [
    "relevance",
    "title",
    "company",
    "location",
    "is_remote",
    "employment_type",
    "salary",
    "posted_date",
    "description",
    "url",
]


def _load_labels() -> dict[str, int]:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def _save_labels(labels: dict[str, int]) -> None:
    LABELS_PATH.write_text(json.dumps(labels, indent=2))


def export_csv() -> None:
    if not SNAPSHOT_PATH.exists():
        print("No snapshot found. Run first: uv run python agent.py --snapshot")
        return

    jobs: list[dict] = json.loads(SNAPSHOT_PATH.read_text())
    labels = _load_labels()

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for job in jobs:
            salary = ""
            if job.get("salary_min"):
                salary = f"{job.get('salary_currency', '')} {job['salary_min']}-{job.get('salary_max', '')}"
            writer.writerow({
                "relevance": labels.get(job["url"], ""),
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "is_remote": job["is_remote"],
                "employment_type": job.get("employment_type", ""),
                "salary": salary,
                "posted_date": job.get("posted_date", ""),
                "description": job.get("description", ""),
                "url": job["url"],
            })

    already = sum(1 for j in jobs if j["url"] in labels)
    print(f"Exported {len(jobs)} jobs to {CSV_PATH}  ({already} already labelled, pre-filled)")
    print("Fill the 'relevance' column (0/1/2), then run: uv run python label_jobs.py --import")


def import_csv() -> None:
    if not CSV_PATH.exists():
        print(f"No CSV found at {CSV_PATH}. Run --export first.")
        return

    labels = _load_labels()
    new_count = 0
    updated_count = 0
    skipped = 0

    with CSV_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["relevance"].strip()
            if raw not in ("0", "1", "2"):
                skipped += 1
                continue
            url = row["url"].strip()
            value = int(raw)
            if url not in labels:
                new_count += 1
            elif labels[url] != value:
                updated_count += 1
            labels[url] = value

    _save_labels(labels)
    print(f"Imported: {new_count} new, {updated_count} updated, {skipped} skipped (blank or invalid).")
    print(f"Total labels in {LABELS_PATH}: {len(labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label jobs for eval")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true", help="Write jobs_for_labelling.csv")
    group.add_argument("--import", dest="import_csv", action="store_true", help="Read labels from CSV into labels.json")
    args = parser.parse_args()

    if args.export:
        export_csv()
    else:
        import_csv()
