#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
# Derived work: a replica of the fn_kmeans function of CMS's published Overall Hospital Quality Star Rating R package (2026 April); that package is a public CMS document and keeps its own terms.
# Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.3 (mechanism and
#     direction of the differences).
# Run:  Rscript mech_fn_kmeans.R in.csv out.csv [seeds.txt]   (called by mech_synthetic.py and mech_worked_example.py)
# Requires: R >= 4.0, base R only (stats::kmeans).

# Verbatim-logic replica of fn_kmeans from CMS R pack program 2
# ("2 - Second Stage_Weighted Average and Categorize Star_2026Apr.R"):
#   stage-1 seeds = medians of the five quintile groups (quantile() default type 7,
#   boundaries at P20/P40/P60/P80, <= on the left);
#   stage 1: kmeans(x, seeds, iter.max=1000)  [Hartigan-Wong default]
#   stage 2: kmeans(x, stage1$centers, iter.max=1000)
#   clusters ordered by center -> star 1..5.
# Input:  CSV with column summary_score (one dataset per file)
# Output: CSV with summary_score, star, plus a JSON-ish stderr dump of seeds/centers.
# Optional 3rd arg: file with 5 alternative seeds (one per line) to seed a single
# kmeans run instead of the two-stage procedure (used for the DP-center check).

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("-h", "--help")) || length(args) < 2) {
  cat("usage: Rscript mech_fn_kmeans.R <in.csv> <out.csv> [seeds.txt]\n",
      "Replica of fn_kmeans from CMS's 2026 R pack: two-stage kmeans on the summary_score\n",
      "column of <in.csv>, stars written to <out.csv>, seeds and centers to stderr; an\n",
      "optional <seeds.txt> (five values, one per line) seeds a single kmeans run instead.\n",
      "Exit codes: 0 success; 2 on a missing argument or input file.\n", sep = "")
  quit(save = "no", status = if (any(args %in% c("-h", "--help"))) 0 else 2)
}
if (!file.exists(args[1])) {
  cat("mech_fn_kmeans.R: missing input file", args[1], "\n", file = stderr())
  quit(save = "no", status = 2)
}

infile <- args[1]; outfile <- args[2]
seedfile <- if (length(args) >= 3) args[3] else NA

d <- read.csv(infile)
x <- d$summary_score

if (!is.na(seedfile)) {
  seeds <- scan(seedfile, quiet = TRUE)
  km <- kmeans(x, matrix(seeds, ncol = 1), iter.max = 1000)
  ord <- order(km$centers)
  star <- match(km$cluster, ord)
  write.csv(data.frame(summary_score = d$summary_score, star = star),
            outfile, row.names = FALSE)
  cat(sprintf("SEEDED centers: %s | iter=%d ifault=%d\n",
      paste(sprintf("%.6f", sort(km$centers)), collapse = ","), km$iter, km$ifault),
      file = stderr())
  quit(save = "no")
}

P <- quantile(x, c(.2, .4, .6, .8))            # default type 7, as in the pack
grp <- ifelse(x <= P[1], 1L, ifelse(x <= P[2], 2L, ifelse(x <= P[3], 3L,
        ifelse(x <= P[4], 4L, 5L))))
seeds <- sapply(1:5, function(g) median(x[grp == g]))

km1 <- kmeans(x, seeds, iter.max = 1000)       # Hartigan-Wong (default)
km2 <- kmeans(x, km1$centers, iter.max = 1000)

ord <- order(km2$centers)
star <- match(km2$cluster, ord)
ord1 <- order(km1$centers)
star1 <- match(km1$cluster, ord1)

write.csv(data.frame(summary_score = d$summary_score, star = star),
          outfile, row.names = FALSE)

cat(sprintf("seeds(quintile medians): %s\n", paste(sprintf("%.6f", seeds), collapse = ",")),
    file = stderr())
cat(sprintf("stage1 centers: %s | iter=%d ifault=%d tot.withinss=%.10f\n",
    paste(sprintf("%.6f", sort(km1$centers)), collapse = ","), km1$iter, km1$ifault,
    km1$tot.withinss), file = stderr())
cat(sprintf("stage2 centers: %s | iter=%d ifault=%d tot.withinss=%.10f\n",
    paste(sprintf("%.6f", sort(km2$centers)), collapse = ","), km2$iter, km2$ifault,
    km2$tot.withinss), file = stderr())
cat(sprintf("stage2 vs stage1 label changes: %d\n", sum(star != star1)), file = stderr())
