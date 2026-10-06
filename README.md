# healthstars replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23195076.svg)](https://doi.org/10.5281/zenodo.23195076)

Replication package for *Stars Misaligned: Medicare Star Ratings And Exact
Clustering* (Matthew D. Schwartz, 2026) and its online appendix. It holds the
code, the verification records and the parsed public inputs behind every
number in both documents.

Most raw CMS downloads (about 1.2 GB) are not included. The exception is the hospital program's inputs under `hospital/data/raw/` and `hospital/work/input/` (CMS's star-rating input files and the published SAS and R programs, unmodified, about 17 MB), which CMS's QualityNet site does not serve to scripted downloads; they are included so the hospital replay runs as shipped. `CMS_RAW_PINS.json`
lists each one by path, byte count, sha256 and source URL; re-fetch them from
those URLs into the listed paths before running the scripts that read them.
Everything derived from them (`data/parsed/`, `spec/`, the hospital R and SAS
replay outputs, the survey census files) is included, so every script that
starts from a derived file runs as shipped.

## Quick check

`evaluator/stars-evaluate.py` is stand-alone (Python 3, standard library only).
It reruns Fisher's exact dynamic program in rational arithmetic on all 123
Medicare Advantage score lists in `evaluator/ma-cutpoint-rows.json` and checks
the published grouping's sum of squares, the optimal sum of squares, the gap,
the uniqueness of the optimum and the count of changed stars against the
recorded values:

    cd evaluator && python3 stars-evaluate.py --full

## Layout

| path | what |
|---|---|
| `src/` | the Medicare Advantage pipeline: score-list reconstruction, Ward's method, the exact optimum (`dp_kmeans.py`), the guardrail, rounding and aggregation, the bonus-status census and dollar estimate, the resampled rerun at sampled input orders, and the additional checks; also `hospital_kmeans.py`, the one-dimensional k-means used in the hospital replay |
| `runs/` | the Medicare Advantage verification records (JSON) that the article and appendix cite, one per computation or check |
| `data/parsed/` | CSV tables parsed from the CMS data-table releases (scores, stars, cut points, summary ratings); `INVENTORY.md` lists them with hashes |
| `spec/` | the CMS Technical Notes transcribed to JSON per star year: clustering rules, measures, aggregation, and the published cut points used as validation targets |
| `hospital/` | the Overall Hospital Quality Star Rating replay: `work/driver.R` runs CMS's published 2026 R program (re-fetched from its pin); `work/sas_replay.py` replays the 2021-2025 SAS programs with the one k-means routine rewritten from its documentation; `work/exact_census*.py` compare the published stars with the exact optimum; `runs/` holds the records |
| `stars_census/` | the patient-experience survey programs (HCAHPS, ICH CAHPS): the census of published groupings against the exact optimum and the Ward replays |
| `figbuild/` | the figure builders for the article's exhibit 3 and the appendix figures, which read the records above |
| `scripts/` | `star_migrations.py`, the data file behind the score-line figures |
| `evaluator/` | the stand-alone check described above |
| `exhibits/` | `exhibit1_by_year.json`, the per-year sums behind the article's exhibit 1 |

## Requirements

Python 3.10 or later with `numpy` and `scipy` (Ward's method), `matplotlib`
(figures), `pandas` and `openpyxl` (parsing), and `pyreadstat` (reads CMS's
`.sas7bdat` hospital input files). R 4.x is needed only to run CMS's own
published 2026 hospital program through `hospital/work/driver.R` and for the
optional cross-check against the CRAN package `Ckmeans.1d.dp`
(`src/crosscheck_ckmeans.py`).

## Regenerating a record

The two survey Ward replays (`stars_census/legc_ich/ich_ward_replay.py` and
`stars_census/legb_hcahps/hcahps_ward_replay.py`) and the resampled-rerun
scripts in `src/` (`enclosure_census_docsonly.py`, `collapse_*.py`,
`group_assignment_candidates.py`) are long-running; each offers a `pilot`
mode or a `[workers]` argument. Run `within_groups_refinal.py` after the
resampled rerun, since it applies the guardrail step to the rerun's records.

One builder reads a raw CMS file that is pinned rather than shipped:
`figbuild/fig_allbins_cahps_ha.py` and the HCAHPS census scripts need the
HCAHPS hospital files under `stars_census/legb_hcahps/data/` (current and the
four archived refreshes), and `hospital/work/coverage_split.py` needs the Care
Compare snapshot zips; fetch them from `CMS_RAW_PINS.json` first. Every other
script runs from the files included.

Each script's docstring gives its inputs, outputs and run line. For example
`cd src && python3 exact_gaps_docsonly.py` rebuilds
`runs/exact_gaps_fullsample_docsonly.json`, the record behind the article's
exhibit 1 panel A (104 of 123 computations miss the true minimum). A script
writes its record to the path named in its docstring, so copy the package
before regenerating if the stored records are to be kept for comparison.

## License and attribution

This package is released under the MIT License (LICENSE), Copyright (c) 2026
Anthropic, PBC. Created by Matthew D. Schwartz; code written by Claude
(Anthropic) under his supervision. This is not an officially supported
Anthropic product; it is maintained by Matthew D. Schwartz
(https://www.bootloops.ai). The data, text and figures are released under
CC BY 4.0. CMS's published programs and data files are public documents and
keep their own terms; they are referenced by pin, not included.

## Citation

Cite the article: M. D. Schwartz, "Stars Misaligned: Medicare Star Ratings
And Exact Clustering" (2026). The archived copy of this package is
M. D. Schwartz, healthstars replication package v1.0, Zenodo (2026),
doi:10.5281/zenodo.23195076 (concept DOI for all versions:
10.5281/zenodo.23195075). The paper's web page is
https://bootloops.ai/summaries/health-stars.html.
