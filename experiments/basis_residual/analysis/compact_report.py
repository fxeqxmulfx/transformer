"""Keep complete class histories locally and retain every aggregate in the report.

The frozen reporter runs first and opens final test only after learning closes.
This storage reduction changes no prediction, score, step, stopping decision or
endpoint class breakdown. It omits repeated per-class tables from history only.
The complete original report remains under runs with its content hash.
"""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from protocol import sha, write


def compact(here=HERE):
    path = here / "results.json"
    result = json.loads(path.read_text())
    if "storage" in result:
        raise ValueError("The report has already been reduced")
    original_hash = sha(path)
    retained = here / "runs" / f"full-results-{original_hash}.json"
    for case in result["cases"]:
        for row in case["history"]:
            for value in row.values():
                if isinstance(value, dict):
                    value.pop("classes", None)
    result["storage"] = {
        "full_result_sha256": original_hash, "full_result_path": str(retained.relative_to(here)),
        "reduction_source_sha256": sha(Path(__file__)),
        "change": "Remove only per-class tables from history; preserve every aggregate and all endpoint class tables."}
    if retained.exists() and sha(retained) != original_hash:
        raise ValueError("The retained full report has changed")
    path.replace(retained)
    write(path, result)
    print(f"Retained complete report {retained.name}; every history aggregate remains in results.json", flush=True)


if __name__ == "__main__":
    compact()
