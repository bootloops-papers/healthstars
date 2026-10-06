# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
# Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.1 and Appendix C.1
#     (rerunning CMS's published 2026 R code reproduces every published star).
# Run:  cd hospital/work && Rscript driver.R   (the pinned R pack unzipped into 'R pack/', the input file in input/)
# Requires: R >= 4.0 and the packages CMS's 2026 R pack loads; the R pack is re-fetched from the pinned R_Pack_2026Apr.zip (see README).

# Thin driver for the CMS 2026 R pack.
# The three CMS programs are sourced byte-unmodified from "R pack/".
# Program 0 hardcodes two Windows paths (Yale storage + a OneDrive R-output dir)
# and begins with rm(list=ls(all=TRUE)), so plain variable injection is impossible.
# Instead the driver attaches an environment carrying a file.path() shim: any path rooted at
# either Windows location is remapped to local sandbox dirs. Attached environments
# survive rm(list=ls()), and the shim delegates to base::file.path otherwise.

if (any(commandArgs(trailingOnly = TRUE) %in% c("-h", "--help"))) {
  cat("driver.R: run CMS's 2026 hospital star-rating R pack byte-unmodified and write\n",
      "R_output/Star_2026apr.csv and R_output/Star_precap_2026apr.csv.\n",
      "Run from hospital/work with the pinned R pack unzipped into 'R pack/' and the\n",
      "input file at input/stars_alldata_2026apr.csv:\n",
      "    Rscript driver.R\n",
      "Exit codes: 0 success; 2 when the R pack or the input file is missing.\n", sep = "")
  quit(save = "no", status = 0)
}
if (!file.exists("R pack/0 - Data and Measure Standardization_2026Apr.R") ||
    !file.exists("input/stars_alldata_2026apr.csv")) {
  cat("driver.R: missing input: 'R pack/' (unzip the pinned R_Pack_2026Apr.zip here) or",
      "input/stars_alldata_2026apr.csv; both are listed with URL and sha256 in CMS_RAW_PINS.json\n",
      file = stderr())
  quit(save = "no", status = 2)
}

INPUT_DIR  <- "input"   # run from hospital/work (paths record-relative)
OUTPUT_DIR <- "R_output"
dir.create(INPUT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

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

cat("STAR TABLE (post safety cap):\n"); print(table(RESULTS$star, useNA = "ifany"))
cat("STAR TABLE (pre safety cap):\n");  print(table(Star_$star, useNA = "ifany"))

# Driver-side extra output (CMS code untouched): the pre-cap star table, used
# to compare the raw k-means assignment against the exact 1-D optimum.
write.csv(Star_, "R_output/Star_precap_2026apr.csv",
          row.names = FALSE)  # literal path: OUTPUT_DIR was rm()'d by program 0
