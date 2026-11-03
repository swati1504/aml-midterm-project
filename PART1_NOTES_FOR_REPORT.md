# Part 1: notes for the write-up

Everything here is reproducible from `midterm/part1.ipynb`. Raw numbers are in `midterm/outputs/*.csv`.
Suggested budget: about 2 of the 5 pages for Part 1.

## One-paragraph abstract (Part 1)
We fill in the 1,014 missing stopping-reason labels using only 157 labeled trials. We compare supervised baselines
(one-vs-rest logistic regression, kNN), unsupervised cluster-then-label methods (K-Means, GMM, HDBSCAN/DBSCAN)
and semi-supervised methods (Label Spreading, self-training, teacher-student augmentation, multi-teacher consensus,
GMM-posterior features), across three preprocessing options and an eight-block feature ablation. Pure clustering
transfers labels poorly (≤ 0.84 sample-F1). A logistic-regression student trained on labeled + pseudo-labeled data
is best. Our final model, an ensemble of self-training and consensus-augmented logistic regression, reaches
**0.900 val sample-F1** (95% bootstrap CI 0.84–0.95) and **0.787 val macro-F1**, versus 0.870 / 0.762 for the
supervised baseline.

## Methods (details graders asked for)
- **Features:** the 150 reason embeddings, z-scored using train statistics (`std`). For distance-based methods, also
  L2-normalised (`l2`, cosine geometry). PCA (20/50 dims) was tested and rejected for the linear models.
- **Decision rule:** predict every reason with p ≥ 0.5; if none, predict the arg-max (every trial has ≥ 1 reason).
- **Model selection:** 5×5 repeated K-fold CV on the 235 labeled trials (157 train + 78 val), with the 1,014 unlabeled
  trials always available to the semi-supervised methods. We also report the val score of models trained on the 157 only.
  Final submission: refit on all 235.
- **Logistic regression (OvR):** L2 penalty, C = 1 (swept 0.1–30), `class_weight='balanced'`, lbfgs, max_iter 3000.
- **K-Means cluster-then-label:** fit on all 1,171 train embeddings (labeled + unlabeled); k-means++ init, n_init = 4–10,
  tol 1e-4, max 300 iterations (converged in 5–20). Cluster label = Laplace-smoothed mean of labeled members,
  (Σy + α·prior)/(n + α), α = 1. k swept 5–150; best k = 50.
- **GMM:** EM, k-means++ init, 3 restarts, tol 1e-4 on the lower bound, max 500 iterations, reg_covar 1e-3. All runs converged
  (≤ 71 iterations). Covariance ∈ {spherical, diag} in 150-d, {diag, tied, full} in PCA-20; k ∈ {2…40}; BIC reported.
  P(reason | x) = Σ_k P(k | x) · smoothed rate_k.
- **HDBSCAN** (min_cluster_size 5–40) / **DBSCAN** (cosine, eps 0.2–0.5, min_samples 5): noise points and clusters with
  no labels fall back to kNN (k = 10); clustered points blend cluster rate and kNN 50/50.
- **Label Spreading:** kNN graph (k = 10), α = 0.2, one binary model per reason, L2 features.
- **Self-training:** LR; 5 rounds; add unlabeled trials with p ≥ 0.9 or ≤ 0.1 on *all five* reasons (806 of 1,014 added by the last round).
- **Augmented LR ← teacher:** the teacher pseudo-labels all unlabeled trials; the student LR trains on true + pseudo labels
  with per-label weight 0.5·|2p − 1|.
- **Consensus-augmented LR:** teachers = LR, Label Spreading, kNN. Keep unlabeled trials where all three predict the identical
  label set (72.7% of them), weight 0.5, retrain LR.
- **Final:** probability average of self-training LR and consensus LR.
- **Prior-shift EM** (Saerens et al., 2002), per reason: tested and reported, but not used in the primary submission
  (val-neutral; changes 14/312 test rows). An alternate submission with it is in `outputs/`.

## Results table (sample-F1; CV = 5×5 repeated K-fold mean ± sd)
| Method | CV sample-F1 | val sample-F1 | val macro-F1 |
|---|---|---|---|
| [sup] LogReg OvR (balanced) | 0.895 ± 0.008 | 0.870 | 0.762 |
| [sup] kNN k=10 (cosine) | 0.877 ± 0.008 | 0.846 | 0.594 |
| [unsup] K-Means k=50 → label | 0.835 ± 0.014 | 0.799 | 0.535 |
| [unsup] GMM diag k=25 → label | 0.787 ± 0.011 | 0.774 | 0.621 |
| [unsup] HDBSCAN → label (+kNN) | 0.886 ± 0.008 | 0.859 | 0.595 |
| [semi] Label Spreading | 0.850 ± 0.007 | 0.859 | 0.601 |
| [semi] LR on emb + GMM posteriors | 0.899 ± 0.006 | 0.868 | 0.715 |
| [semi] Augmented LR ← Label Spreading | 0.892 ± 0.005 | 0.891 | 0.730 |
| [semi] Self-training LR | 0.898 ± 0.010 | 0.897 | 0.789 |
| [semi] Consensus-augmented LR | **0.903 ± 0.007** | 0.882 | 0.765 |
| **FINAL: Ens(Self-training + Consensus)** | 0.898 ± 0.011 | **0.900** | **0.787** |

Final model, val per-reason F1: Safety 0.89 · Funding 0.67 · Efficacy 0.83 · Recruitment 0.98 · Other 0.57.
The full table (15 methods, all per-reason F1) is `outputs/part1_results_summary.csv`.

## Insights worth writing up (ranked)
1. **Clustering ≠ labeling.** K-Means and GMM clusters don't line up with reasons. Purity rises with k (0.65 → 0.95) but the share of
   clusters containing any labeled trial falls (100% → 49%), so F1 peaks at k ≈ 50 (0.84), well below LR (0.895). The 157-label budget is the bottleneck.
2. **BIC-optimal ≠ label-optimal.** BIC picks k = 5 (full), 15 (diag), 40 (spherical/tied), yet label transfer improves with larger k for every covariance type.
3. **PCA cuts both ways.** It hurts LR (0.894 → 0.854 at 20 dims), because rare-reason directions are low-variance, but it helps kNN-graph / clustering
   methods (Label Spreading 0.837 → 0.872), because removing noisy dimensions makes distances more meaningful.
4. **Density methods label 45–79% of trials as noise** in 150-d. Their F1 is basically the kNN fallback.
5. **Semi-supervised gains are real but modest** (+0.5–1 pt CV, +1–3 pts val over LR). The learning curve shows self-training ≥ LR at every label budget
   (15 → 157). Consensus only pays off at the full 157 labels.
6. **Consensus pseudo-labels are biased toward easy Recruitment cases:** among the 73% of trials where the teachers agree, only 0.3% are Funding
   and 2.4% are Other. Re-weighting rare positives (balanced consensus) changed nothing, because there were almost none to re-weight.
7. **Why including everything fails (feature ablation):** no extra block (start_year, disease/criteria TF-IDF, ICD chapters, Morgan fingerprints,
   Reactome pathways, drug actions, has-drug flag) beats embeddings-only beyond CV noise. All blocks together (+322 dims) drop val from 0.870 to 0.850.
   The non-embedding features *alone* score only 0.53 sample-F1. Reasons: they describe trial *design*, not why it stopped; 157 labels can't support
   472 dims; drug features are missing for 44% of trials.
8. **Time shift:** labeled trials are early (median 2001), test trials late (2010–16). The model's raw test predictions already contain more Safety
   (16% vs a 10% training prior) and Other, less Recruitment. Reason mix over time: Safety 2% → 16%, Recruitment 95% → 59%. EM prior correction was
   val-neutral, so it was left out of the primary submission (good honest-negative result).
9. Thresholds: per-label tuned thresholds give no gain. ~93% of trials have one reason, so the ≥ 1-label arg-max rule does most of the work and the threshold only governs the occasional second label.

## Figures (`midterm/figures/`); suggest picking 3–4 for the report
- `fig5_method_comparison.png` (main result, must-have)
- `fig2_tsne_by_reason.png` (why embeddings work)
- `fig6_learning_curve.png` (when unlabeled data helps)
- `fig3_kmeans_k_sweep.png` or `fig4_gmm_bic.png` (unsupervised insight)
- `fig8_prior_shift.png` / `fig9_reason_mix_over_time.png` (time shift)
- `fig1_start_year_shift.png`, `fig7_threshold.png` (optional)

## References
Saerens, Latinne & Decaestecker 2002 (prior-shift EM) · Zhou et al. 2004 (Label Spreading) · Yarowsky 1995 (self-training) ·
Blum & Mitchell 1998 (co-training) · Campello et al. 2013 (HDBSCAN) · Ester et al. 1996 (DBSCAN) · Fu et al. 2022 (HINT) ·
Pedregosa et al. 2011 (scikit-learn) · Rogers & Hahn 2010 (ECFP/Morgan fingerprints) · RDKit.
