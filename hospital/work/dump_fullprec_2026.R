# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
# Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.1 and Appendix C.1
#     (the binary floating-point scores of the R code, 2,978 of 3,203 differing
#     from the fifteen-digit decimals in the last bits).
# Run:  cd hospital/work && Rscript dump_fullprec_2026.R   (the R pack in 'R pack/' and the input file in input/, as for driver.R)
# Requires: R >= 4.0 and the packages CMS's 2026 R pack loads; the R pack is re-fetched from the pinned R_Pack_2026Apr.zip (see README).

# Full-precision dump driver for the CMS 2026 R pack.
# Same shim mechanism as driver.R: the three CMS programs are sourced byte-unmodified;
# only the two hardcoded Windows paths are remapped. After the pipeline runs, the script
# dumps the post-program-2 global `qq` (= Star_ + Safe_q + star_p3, i.e. the cap worksheet
# before the columns are dropped into RESULTS) with summary_score and
# Std_Outcomes_Safety_score printed at %.17g so the IEEE doubles round-trip exactly.
# A round-trip check (as.numeric(sprintf) == original) is asserted in-run.

if (any(commandArgs(trailingOnly = TRUE) %in% c("-h", "--help"))) {
  cat("dump_fullprec_2026.R: run CMS's 2026 R pack byte-unmodified and dump the post-program-2\n",
      "worksheet with summary_score and Std_Outcomes_Safety_score at %.17g to\n",
      "R_output/fullprec_2026apr.csv.\n",
      "Run from hospital/work with the pinned R pack unzipped into 'R pack/' and the\n",
      "input file at input/stars_alldata_2026apr.csv:\n",
      "    Rscript dump_fullprec_2026.R\n",
      "Exit codes: 0 success; 2 when the R pack or the input file is missing.\n", sep = "")
  quit(save = "no", status = 0)
}
if (!file.exists("R pack/0 - Data and Measure Standardization_2026Apr.R") ||
    !file.exists("input/stars_alldata_2026apr.csv")) {
  cat("dump_fullprec_2026.R: missing input: 'R pack/' (unzip the pinned R_Pack_2026Apr.zip here) or",
      "input/stars_alldata_2026apr.csv; both are listed with URL and sha256 in CMS_RAW_PINS.json\n",
      file = stderr())
  quit(save = "no", status = 2)
}

INPUT_DIR  <- "input"   # run from hospital/work (paths record-relative)
OUTPUT_DIR <- "R_output"

shim <- new.env()
shim$file.path <- local({
  win_in  <- "\\\\storage.yale.edu\\home\\CORE_Analysis-CC1022-MEDINT\\HC2526\\star"
  win_out <- "C:\\Users\\lq28\\OneDrive - Yale University\\Yale Box\\CORE\\Star Rating_Remote\\2026 Apr\\R pack\\R output"
  ind <- INPUT_DIR; outd <- OUTPUT_DIR
  function(...) {
    args <- list(...)
    if (length(args) >= 1 && is.character(args[[1]]) && length(args[[1]]) == 1) {
      if (identical(args[[1]], win_in))  args[[1]] <- ind
      if (identical(args[[1]], win_out)) args[[1]] <- outd
    }
    do.call(base::file.path, args)
  }
})
attach(shim, name = "cms_path_shim")

# setwd: run this script from hospital/work
source("R pack/0 - Data and Measure Standardization_2026Apr.R", echo = FALSE)
source("R pack/1 - First stage_Simple Average of Measure Scores_2026Apr.R", echo = FALSE)
source("R pack/2 - Second Stage_Weighted Average and Categorize Star_2026Apr.R", echo = FALSE)

# qq survives program 2 in the global env: Star_ columns + Safe_q + star_p3.
stopifnot(exists("qq"), all(c("Safe_q", "star_p3", "star") %in% names(qq)))

fmt17 <- function(x) ifelse(is.na(x), "NA", sprintf("%.17g", x))

# Round-trip assertion: %.17g strings re-parse to the identical doubles.
rt <- function(x) {
  ok <- is.na(x) | (as.numeric(sprintf("%.17g", x)) == x)
  all(ok)
}
stopifnot(rt(qq$summary_score), rt(qq$Std_Outcomes_Safety_score))
cat("ROUNDTRIP_OK summary_score and Std_Outcomes_Safety_score\n")

dump <- data.frame(
  PROVIDER_ID              = qq$PROVIDER_ID,
  report_indicator         = qq$report_indicator,
  cnt_grp                  = qq$cnt_grp,
  Total_measure_group_cnt  = qq$Total_measure_group_cnt,
  Outcomes_Safety_cnt      = qq$Outcomes_Safety_cnt,
  Safe_q                   = qq$Safe_q,
  star_precap              = qq$star,
  star_postcap             = qq$star_p3,
  summary_score_17g        = fmt17(qq$summary_score),
  safety_score_17g         = fmt17(qq$Std_Outcomes_Safety_score),
  stringsAsFactors = FALSE
)
dump <- dump[order(dump$PROVIDER_ID), ]
write.csv(dump, "R_output/fullprec_2026apr.csv",
          row.names = FALSE)

cat("PRECAP  TABLE:\n"); print(table(qq$star,    useNA = "ifany"))
cat("POSTCAP TABLE:\n"); print(table(qq$star_p3, useNA = "ifany"))
cat("CAPPED_5_TO_4:", sum(qq$star == 5 & qq$star_p3 == 4, na.rm = TRUE), "\n")
cat("DUMP_ROWS:", nrow(dump), "\n")
