# healthstars v1.0

Replication package for *Stars Misaligned: Medicare Star Ratings And Exact
Clustering* (Matthew D. Schwartz, 2026) and its online appendix: the code,
the verification records and the parsed public inputs behind every number in
both documents.

- `evaluator/stars-evaluate.py --full` reruns the exact optimum on all 123
  Medicare Advantage score lists and checks every recorded value (standard
  library only).
- Every record under `runs/`, `hospital/runs/` and `stars_census/` regenerates
  from the included scripts and inputs; `MANIFEST.sha256` lists every file.
- Raw CMS downloads are pinned by URL and hash in `CMS_RAW_PINS.json`; the
  hospital program's inputs are included.

The archived copy of this release is deposited on Zenodo (DOI in README.md).
