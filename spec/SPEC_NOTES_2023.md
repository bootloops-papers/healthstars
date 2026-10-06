# SPEC_NOTES_2023 — clustering-side specification sources, star year 2023

Companion to `measures_2023.json`, `clustering_2023.json`,
`validation_targets_2023.json`; extends `SPEC_NOTES.md` (2024–2026) back one
year. All quotes verbatim from the primary sources. Built by
`scripts/extract_tn2023.py` + `scripts/build_vt2023.py` +
`scripts/build_clustering2023.py`.

## Sources

| doc | file | version | md5 |
|---|---|---|---|
| 2023 final | `data/raw/sy2023/2023-star-ratings-technical-notes.pdf` (live cms.gov) | Updated 01/19/2023, 203 pp | 57d3b8342cf3b3fde1da601da3b97cab |
| 2023 first release | `data/raw/sy2023/2023technotes-wayback-20221122.pdf` | Updated 09/28/2022, 209 pp | 77e70ccfdb6850b650dff3a4c1a83f74 |
| 2023 first plan preview | `data/raw/sy2023/2023technotes-wayback-20220829.pdf` | first plan preview, Aug 2022 | 0b3d2ce83b54f99ca4b1fc86de36c073 |
| 2023 Cut Point Trend | `data/raw/cut-point-trends.zip :: 2023_Cut_Point_Trend_2022_12_12.pdf` | Updated 10/31/2022 | (zip pinned) |
| 2022 data tables | `data/raw/sy2022/2022-star-ratings-data-table-oct-06-2021.zip` | Oct 06 2021 | e70f55b95fcf5d690e959211573785a2 |
| 2023 Rate Announcement | cms.gov (not included in the record; used only to confirm the guardrail delay) | Apr 2022 | — |

Full source list incl. the truncated-in-archive Oct 2022 wayback capture:
`data/raw/sy2023/SOURCES_TECHNOTES_sy2023.txt`.

## (a) 2023 vs 2024 — the year-over-year methodology diffs that matter

1. **NO TUKEY OUTLIER DELETION IN 2023.** Zero occurrences of
   Tukey/outlier/outer-fence in either 2023 Tech Notes version; no K-5/K-6
   equivalent tables. The clustering input goes straight to mean resampling.
   Tukey deletion begins with the 2024 Star Ratings (2024 change list; also
   the 2025 "Note on References to the 2024 Star Ratings" describes the 2023
   cut points as computed without Tukey deletion). **Consequence: the
   published outlier bounds used to check the 2024–2026 score lists do not
   exist for 2023.** The only published clustering-side numbers for 2023 are
   J-3/J-4 (mean-resampling thresholds — dependent on the random group
   assignment, display-rounded) and the final (guardrail-applied) cut points.

2. **2023 is the FIRST guardrail year.** 2023 change (a); CMS-1744-IFC
   delayed §§422.166(a)(2)(i)/423.186(a)(2)(i) guardrails to the 2023 Star
   Ratings (2023 Rate Announcement p.79); the eCFR text says "Effective for
   the Star Ratings issued in October 2022 and subsequent years".
   Guardrail exemptions in 2023: improvement measures (C25/D04), new
   measures ≤3 years — C12 (respecified CBP, weight 1, change (c)) and D07
   (MPF Price Accuracy, treated as new through 2024 per the 2025 change
   note; 2023-empirically exempt: final == J-4 both org types despite
   |pre − 2022 base| > 5 on two PDP boundaries). The "returning measures"
   sentence (2025+) does not exist yet.

3. **Guardrail BASE convention (pinned as far as public data allows).** The
   2023 Tech Notes say only "the prior year's cut point" and publish **no
   basis table** (2024 published its rerun basis as K-7/K-8). Exact
   display-precision clip arithmetic against the **published 2022 cut
   points** (2023 Cut Point Trend 2022 rows == Oct 06 2021 data tables:
   240/240 star-cells exact across all 48 compared splits — the base is
   fully double-sourced (runs/tn2023_release_diff.json,
   base22_double_source), and any post-release 2022 revision left cut
   points untouched) reproduces the published finals for **26 of 32
   cap-tested guarded splits** (35 guarded rows total, incl. the 3
   restricted-range complaints splits tested separately; e.g. C08 binds at
   base+5 on all four boundaries, C26 on three, C07 on two, D01/MA-PD on
   two, C24/D03 at base−5). It is **falsified on 9 splits** — C01, C02,
   C10, C28 (ALL), D09/MA-PD, D11/MA-PD, and the complaints splits C23/ALL,
   D02/MA-PD, D02/PDP — **11 boundaries** (7 cap-test + 4 restricted-range),
   9 of the 11 at the 1|2-star boundary (counts from
   spec/validation_targets_2023.json final_vs_pre_guardrail_check.tallies).
   Two cells are impossible under ANY clip against the published base (C10
   1|2-star: pre 86, base 82, final 80; C28 2|3-star: pre 77, base 61,
   final 59 — final on the far side of the base). Every falsifying cell
   (11/11) is MORE generous than the
   model. Inference (labeled as such in the spec): the operative base was an
   **unpublished recomputation of the 2022 cut points** — the same rerun
   convention 2024 later made explicit — consistent with non-substituted
   (pre-COVID-better-of) scores; a 1-realization replay of the published
   2022 scores lands exactly on the implied base for C01 (38) and within 1
   for D09/MA-PD, but far off for C10/D11 (mechanism not determined). The
   replay therefore treats 2023 guardrails the same way as the other years:
   published prior-year FINAL boundaries as base, with the falsified subset
   flagged.

4. **Restricted-range cap definition changed INSIDE the 2023 star year.**
   First release (09/28/2022): cap = 5% of the **current** year's (max −
   min) score range — in both the Methodology body and Attachment J. The
   01/19/2023 update changed both passages to the **prior** year's range
   (the 2024+ wording), and changed the C23/D02(MA-PD) 1–2 star finals
   1.53 → 1.52. **The published data tables (Oct 04 2022 — the only vintage
   ever posted; single wayback content digest) still carry 1.53.** The
   published stars use 1.53. Neither value is reproducible
   from public data (published base 1.14 + published-range caps → 1.27–1.28)
   — the complaints splits belong to the falsified-base subset above.
   Impact check [exact]: no MA-side contract has a published C23/D02 score
   in (1.52, 1.53] — the discrepancy flips no published measure star.
   The 2025-era "excluding outer fence outliers" qualifier does not exist in
   2023 (no Tukey).

5. **Disaster rules 2023** (COVID-era): thresholds 25%/60% as in 2024/2025.
   Clustering exclusion: "the measure scores for contracts with 60% or more
   of their enrollment affected by a disaster are excluded from creating
   those cut points. **For the 2023 Star Ratings only, we will not exclude
   contracts with 60% or more of their enrollees living in FEMA-designated
   Individual Assistance areas from calculation of cut points for the 3
   HEDIS-HOS measures**" (Monitoring Physical Activity C04, Reducing the
   Risk of Falling C13, Improving Bladder Control C14 — change (e)).
   Reward-factor exclusion names **2021** disasters; summary CSV columns are
   "2020 Disaster %" / "2021 Disaster %" (recent = 2021). The 25% rules are
   score substitutions ("higher of 2022/2023" per measure family), not
   clustering-input exclusions. HEDIS-HOS 25% adjustments key on **2020**
   disasters (2-year HOS lag). Prior-year context: 2022 waived the 60%
   clustering exclusion ENTIRELY ("For the 2022 Star Ratings only...").

6. **Measure registry**: 40 measures (28 C + 12 D) vs 42 in 2024.
   2023→2024 name map (verified, `measures_2023.json.measure_id_2024`):
   all 12 D IDs unchanged; C01–C09 unchanged; C10 (Diabetes Care – Kidney
   Disease Monitoring) retired after 2023 (maps to null); C11–C15 shift
   DOWN one (2023 C11 Blood Sugar→2024 C10, C12 CBP→C11, C13 Falling→C12,
   C14 Bladder→C13, C15 MRP→C14); C16 Statin Therapy stays C16; C17–C28
   shift UP two (C17 Getting Needed Care→C19 ... C28 Call Center→C30),
   because 2024 inserts C15 Plan All-Cause Readmissions, C17 Transitions of
   Care, C18 Follow-up after ED (2024-new). Weights: patient
   experience/complaints/access measures moved to weight 4 in 2023 (change
   (b)); SUPD recategorized process/weight 1 (change (d)); C12 CBP weight 1
   (new). 2023 carries NQF # (no CMIT — CMIT IDs first appear in 2024).
   Retired into 2023: Rheumatoid Arthritis Management.

7. **Everything verbatim-identical to 2024** (verified in the extracted
   passages): `proc surveyselect data=inclusterdat groups=10 seed=8675309
   out=inclusterdat_random;` + the same narrative incl. "the input dataset,
   inclusterdat, is the list of contracts without missing, flagged, excluded
   by disaster rules or voluntary contract scores"; PROC
   DISTANCE/CLUSTER/TREE code and options; squared-distance Ward; ncl=NSTARS
   with 5 default and combine-equal-range rule; improvement split NSTARS=3
   (≥0)/2 (<0); min/max per-group cut-point convention; mean over 10
   groups; "The lower limit of each cluster becomes the cut point" (2026
   adds the higher-is-better qualifier); replication caveat sentence (2024
   comma variant); "traditional rounding" language.

## (b) First release (09/28/2022) vs final (01/19/2023) — complete catalog

Normalized line diff (runs/tn2023_release_diff.json): **200 hunks**, 89 with
digit-token changes (mostly ToC page numbers / pagination shifts between
the 209 pp and 203 pp renders / formula image-vs-text rendering). The six
substantive changes, each located in the diff record (substantive_catalog):

1. **C23 final cut points**: 1-star "> 1.53" → "> 1.52"; 2-star upper
   likewise (Measure Details).
2. **D02 MA-PD final cut points**: same 1.53 → 1.52 change.
   The change log entry names "C24 and D02 (MAPD cut points only)"; in the
   diff record the cut-point cells that change numerically are C23's and
   D02(MA-PD)'s, while C24's cells differ only in spacing ("<=7%" →
   "<= 7 %").
3. **Restricted-range cap definition**: "current year's" → "prior year's"
   (twice: Methodology body + Attachment J).
4. **Table J-3 direction symbols**: C23 5-star ">= 0.19" → "<= 0.19";
   C24 5-star ">=7%" → "<= 7 %" (both measures lower-is-better per their
   General Trend fields).
5. **Table 4** (minimum measure scores for an improvement-measure rating),
   Part D row: "6 of 11" → "5 of 11" in **four columns** (one the starred
   variant) — the change is not described in the change log entry.
6. CAHPS J-5/J-6 star-assignment table cells shifted in two rows
   (rendering of the same rules; column count difference in the text layer
   only).

The trend document (10/31/2022) predates the change and carries 1.53; its
D12/PDP 2023 4-star cell reads ">= 84 % to < 84 %" — an empty interval as
printed (both Tech Notes and the data table read "< 86 %").

## (c) Why the 2023 checks differ

- **Score-list membership**: 2023 publishes no outlier bounds, so
  membership cannot be checked exactly; the pre-guardrail J-3/J-4
  thresholds are display-rounded means over the random group assignment.
  The 2023 score list follows the documented rules only (exclude 1876 Cost
  "voluntary" contracts and ≥60% 2021-disaster contracts, with the
  C04/C13/C14 waiver) and is labeled "hypothesis"; no 2023 target can be
  labeled "exact".
- **Gap computation**: the two-arm SSQ gap is exact given the stated
  input; 2023 rows carry membership label "hypothesis".
- **2023→2024 comparison**: the 2023 replay runs with `tukey=False`; 2023
  guardrails use the published 2022 finals as base, with the caveat in
  (a)3; the 2024 side reuses the documented-rules-only 2024 results.
