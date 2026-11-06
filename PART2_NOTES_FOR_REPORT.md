# Part 2: notes for the write-up

Everything here can be reproduced from `midterm/part2.ipynb`, which runs top to bottom in about 50 min on a laptop CPU.
The raw numbers are in `midterm/outputs/p2_*.csv`.
Suggested budget: about 2.5 of the 5 pages for Part 2.

## Checklist: what the spec requires for Part 2, and where it is below
| Required by the spec | Where |
|---|---|
| Binary classifier per phase (or one model that uses phase) | Methods → Models; Results table (per-phase vs pooled both reported) |
| Val F1, ROC-AUC, PR-AUC **for each phase** (label = 1 positive; F1 from binary predictions) | "Per phase, original → improved" table |
| Error analysis: unsuccessful val trials matched to Part 1 reasons, phases separate, a "no recorded reason" group | "Error analysis by stopping reason" table |
| **State which reason labels are true vs Part 1 predictions** | Line above that table (27 true, 171 predicted, 254 no reason) + insight 5 |
| For each reason × phase: number of trials and fraction predicted successful | Same table (n, original, improved) |
| Improve the classifier, compare overall metrics **and** per-reason error rates with the original | Results table + per-phase table + error table + significance section |
| Discuss results even where it does not improve | Phase III-targeted attempts (negative) + insight 4 |
| Discuss differences across reason categories, considering category sizes and predicted-label errors | Insights 4 and 5 |
| Evaluate the modification and its **trade-offs** | Threshold trade-off section + insight 10 |
| Discuss the val → test gap | Kaggle section + insights 7–8 (temporal holdout) |
| Reason embeddings / labels not used as inputs | Methods → Data / leakage (first bullet) |
| Kaggle submission (ID, label = P(label = 1)) | Kaggle section |
| Methods detail (initialisation, training, convergence) | Methods (K-Means init, HGB settings, stacker, CV scheme) |

**If space is tight, keep:** the per-phase original-vs-improved table, the error-by-reason table (it can be trimmed to the All-failures / Recruitment / No-reason rows plus a sentence on the small cells), the significance line, the threshold trade-off sentence, the test-gap paragraph, and figures `p2_fig5` and `p2_fig6`.

## One-paragraph abstract (Part 2)
We predict whether each trial meets its primary endpoint, separately for Phases I, II and III.
- **Original model:** a per-phase logistic regression on TF-IDF features of eligibility criteria, disease, drug and ICD text.
  It reaches **0.736 mean val ROC-AUC, 0.714 F1 and 0.589 log loss**.
- **Error analysis:** we link the val failures to the Part 1 stopping reasons. Phase I is near a coin flip (53% of its failures are predicted successful) and Phase III is worse (57%).
  Recruitment failures are caught in Phase II (23% error) but not in Phase III (68%).
- **Targeted improvements:** we add features aimed at these failure modes: eligibility-design features, HINT-style historical outcome rates for each drug, disease, ICD code and target, Tanimoto drug-structure neighbours, and K-Means cluster rates. We also pool the phases.
- **Final model:** a stacked logistic regression over five base models (two logistic regressions, gradient boosting, Multinomial NB, kNN), with per-phase intercepts and a start-year trend.
  It reaches **0.760 ROC-AUC, 0.737 F1, 0.795 PR-AUC and 0.569 log loss**, and cuts the Phase I error on failures from 53% to 39%.
- **Time shift:** a temporal holdout (fit ≤ 2009, test 2010–13) shows the gap to expect on the 2014–21 test set: ROC-AUC falls by about 0.04 and log loss rises.
  We use the holdout to choose how to extrapolate the strongly rising success rate.

## Methods (details)
- **Data / leakage.** One row is one trial in one phase. Multi-phase trials appear once per phase with the same label, so **every CV split is a GroupKFold on `trial_id`**.
  1,504 train rows (19%) are Part 1 trials, and 1,503 of them failed (label mean 0.001 vs 0.683 for the rest). No test trial is in Part 1, so a "trial is in Part 1" flag would be a near-perfect but useless leaked feature, and we never use it.
  Reason embeddings and reason labels are used only for the error analysis.
- **Features.** Every transformer is fit on the training fold only.
  - `tfidf`: TF-IDF of criteria (1–2-grams, min_df 5, max 40k, sublinear tf, English stop words), disease names (1–2-grams), drug names, and ICD-3 codes plus ICD chapters. 44,388 dimensions in total.
  - `counts`: unigram bag of words, for Multinomial NB.
  - `design` (26 features): number of inclusion and exclusion bullets, their log ratio, criteria length, lab-threshold count, age min/max/span, and flags for healthy volunteers, paediatric, refractory/relapsed, metastatic, pregnancy, ECOG/Karnofsky and life expectancy. Also the number of drugs, diseases and ICD codes, placebo arm, combination therapy, has-matched-drug flag, number of matched drugs/targets/pathways, multi-phase flag and oncology flag.
  - `desc`: mean RDKit descriptors of the matched drugs (MolWt, LogP, TPSA, HBD, HBA, RotB, Rings, QED). Missing values are filled with the train median; the has-match flag carries the missingness.
  - `fp`: 1024-bit Morgan fingerprint (radius 2), element-wise max over drugs.
  - `act` / `path`: multi-hot drug actions (17) and Reactome pathways (1,195).
  - `hist` (16 features; HINT idea, Fu et al. 2022): for each drug, disease, ICD-3 code and protein target, the shrunk mean outcome residual against the phase prior, Σr/(n + 10), computed over *other* training trials.
    Aggregated per trial as mean, min, max and log(#prior trials). Out-of-fold (5-fold grouped) on the rows a model is trained on.
  - `tani`: the similarity-weighted outcome residual of the 10 most Tanimoto-similar training trials by fingerprint, plus the top similarity. Out-of-fold.
  - `clrate_k`: K-Means (k-means++, n_init = 4) with k ∈ {10, 25, 50} on the L2-normalised 100-d truncated SVD of criteria TF-IDF; feature = the cluster's shrunk outcome residual, Σr/(n + 20). Out-of-fold.
  - `svd`: the same 100-d criteria SVD, used directly as features.
- **Models.**
  - Logistic regression (L2, lbfgs): per phase, C ∈ {0.1, 0.3, 1, 3}. The pooled version adds a phase one-hot and the standardised dense blocks.
  - Multinomial NB (α ∈ {0.1, 1}).
  - kNN (cosine on TF-IDF, distance-weighted, k ∈ {15, 50, 100}).
  - HistGradientBoosting: pooled over phases with a phase one-hot. Final setting: lr 0.05, 300 iterations, 15 leaves, min_samples_leaf 40, L2 1.0, no early stopping. Swept lr {0.03, 0.05, 0.1}, iterations {150, 300, 600} and leaves {7, 15, 31}.
- **Stacking + calibration.** A logistic-regression meta-learner (C = 1, no global intercept) is trained on **5-fold grouped out-of-fold predictions** of the five base models.
  Its inputs are the logit of each base prediction, the three phase intercepts, and three phase × (start_year − 2006)/5 terms.
  It therefore combines the models, Platt-calibrates them and learns a per-phase logistic year trend in one step.
  For test, the year is capped at the last training year (2013). The cap was chosen on the temporal holdout, where all caps from +0 to +3 years and no cap were within 0.001 log loss.
- **Evaluation.**
  - Val metrics per phase: F1 at threshold 0.5, ROC-AUC, PR-AUC (average precision) and log loss, each averaged over the three phases.
  - Also the 5-fold grouped-CV mean ± sd on train. This is more stable than val, which has only 149 Phase I rows.
  - Final submission: base models refit on train + val; stacker fit on train + val OOF predictions; predictions clipped to [0.02, 0.98].

## Results table (mean over phases; val = outcome_val, CV = 5-fold grouped CV on train)
| Model | val F1 | val ROC-AUC | val PR-AUC | val LogLoss | CV ROC-AUC |
|---|---|---|---|---|---|
| **Original:** LR per-phase, TF-IDF 1–2g, C=0.3 | 0.714 | 0.736 | 0.786 | 0.589 | 0.709 ± 0.018 |
| LR per-phase, TF-IDF unigram | 0.711 | 0.740 | 0.787 | 0.586 | 0.710 ± 0.017 |
| Multinomial NB, BoW, α=1 | 0.695 | 0.689 | 0.706 | **3.620** | 0.684 ± 0.014 |
| kNN cosine, k=50 | 0.703 | 0.709 | 0.760 | 0.609 | 0.692 ± 0.020 |
| LR pooled + phase one-hot, TF-IDF | 0.729 | 0.741 | 0.772 | 0.584 | 0.727 ± 0.018 |
| HGB: phase + design only | 0.695 | 0.674 | 0.719 | 0.633 | 0.656 ± 0.014 |
| HGB: phase + hist only | 0.711 | 0.735 | 0.770 | 0.588 | 0.696 ± 0.018 |
| HGB: all dense (design+hist+tani+desc+act) | 0.732 | 0.736 | 0.769 | 0.586 | 0.704 ± 0.021 |
| HGB: all dense − hist | 0.708 | 0.696 | 0.726 | 0.618 | 0.671 ± 0.010 |
| HGB: all dense + text SVD-100 | 0.723 | 0.755 | 0.784 | 0.576 | **0.729 ± 0.021** |
| LR pooled: TF-IDF + dense, C=0.3 | 0.728 | 0.747 | 0.771 | 0.579 | 0.726 ± 0.016 |
| LR pooled: *everything* (+fp, pathways, actions) | 0.712 | 0.730 | 0.762 | 0.593 | 0.708 ± 0.015 |
| Stack (5 bases), no year | 0.738 | 0.760 | 0.796 | 0.567 | – |
| **Improved: Stack (5 bases) + year trend** | **0.737** | **0.760** | **0.795** | **0.569** | – |

Per phase, original → improved (val):

| Phase | F1 | ROC-AUC | PR-AUC | LogLoss |
|---|---|---|---|---|
| I | 0.734 → 0.771 | 0.727 → 0.758 | 0.795 → 0.790 | 0.611 → 0.584 |
| II | 0.611 → 0.645 | 0.715 → 0.740 | 0.697 → 0.721 | 0.615 → 0.596 |
| III | 0.798 → 0.795 | 0.767 → 0.783 | 0.867 → 0.876 | 0.541 → 0.526 |

**Is the improvement significant?** (`outputs/p2_bootstrap_improvement.csv`; notebook Exp D / Section 9)
- **Cross-fitted CV of the stack** (the stacker is fit on 4 folds of out-of-fold base predictions and scored on the 5th):
  - ROC-AUC 0.709 → 0.738 (+0.029), PR-AUC +0.024 and log loss −0.018: the stack wins in **5/5 folds** on all three.
  - F1 +0.014, winning in 4/5 folds.
  - Per-phase CV AUC: I 0.645 → 0.696, II 0.712 → 0.728, III 0.770 → 0.790.
- **Paired bootstrap on val** (B = 1000, resampling within each phase), improved − original:

| Metric | Mean difference | 95% CI |
|---|---|---|
| ROC-AUC | +0.024 | [0.004, 0.045] |
| Log loss | −0.020 | [−0.034, −0.006] |
| F1 | +0.023 | [0.002, 0.043] |
| PR-AUC | +0.009 | [−0.015, 0.033] (*not significant*) |

Stacker weights (val fit): lr_phase 0.25, lr_pool 0.30, hgb 0.41, kNN 0.09, **MNB 0.00**; year slope is largest for Phase III (0.32).
The full table (42 configurations) is in `outputs/p2_results_summary.csv`, and the per-phase version is in `outputs/p2_results_by_phase.csv`.

## Error analysis by stopping reason (val, label = 0; error = fraction predicted successful at p ≥ 0.5)
Reason source: true labels for trials in Part 1's labelled train/val sets (27 trials); Part 1 *predicted* labels for the other 171 Part 1 trials. 254 failures are not in Part 1 ("No recorded reason").

| Phase | Reason | n | original | improved |
|---|---|---|---|---|
| I | Safety | 8 | 0.62 | 0.50 |
| I | Funding | 4 | 0.50 | 0.00 |
| I | Efficacy | 2 | 1.00 | 1.00 |
| I | Recruitment | 22 | 0.55 | **0.27** |
| I | Other | 5 | 0.80 | 0.80 |
| I | No recorded reason | 28 | 0.46 | 0.39 |
| I | **All failures** | 64 | 0.53 | **0.39** |
| II | Safety | 13 | 0.38 | 0.31 |
| II | Funding | 4 | 0.00 | 0.00 |
| II | Efficacy | 12 | 0.42 | 0.50 |
| II | Recruitment | 86 | 0.23 | 0.21 |
| II | Other | 25 | 0.44 | 0.48 |
| II | No recorded reason | 144 | 0.31 | 0.31 |
| II | **All failures** | 265 | 0.29 | 0.28 |
| III | Safety | 6 | 0.33 | 0.33 |
| III | Funding | 0 | – | – |
| III | Efficacy | 10 | 0.60 | 0.70 |
| III | Recruitment | 22 | 0.68 | 0.68 |
| III | Other | 10 | 0.60 | 0.70 |
| III | No recorded reason | 82 | 0.56 | 0.54 |
| III | **All failures** | 123 | 0.57 | 0.56 |

### Phase III-targeted attempts (Exp D2), both negative
| Model | Phase I val AUC | Phase II val AUC | Phase III val AUC | CV AUC |
|---|---|---|---|---|
| HGB dense + SVD (reference) | 0.755 | 0.738 | 0.771 | 0.729 |
| + prior-phase history (drug/disease record in earlier phases) | 0.758 | 0.729 | 0.766 | 0.730 |
| LR per-phase (original) | 0.727 | 0.715 | 0.767 | 0.709 |
| LR per-phase, class-balanced | 0.726 | 0.715 | 0.767 | 0.709 |

- **Prior-phase history** duplicates what the pooled `hist` encoding already contains, and Phase III drugs have few earlier-phase trials here.
- **Class balancing** moves the operating point without changing the ranking, and it hurts F1 and log loss.
- So Phase III's *ranking* cannot be improved with these inputs; the remaining lever is the threshold (below).

### Threshold trade-off (`outputs/p2_threshold_tradeoff.csv`, figure `p2_fig6`)
Thresholds are chosen per phase on the cross-fitted *training* OOF predictions and then applied to val.

| Rule | Phase | Threshold | F1 | Recall on successes | Error on all failures | Recruitment | Efficacy | No recorded reason |
|---|---|---|---|---|---|---|---|---|
| default 0.5 | III | 0.50 | 0.795 | 0.850 | 0.561 | 0.682 | 0.700 | 0.537 |
| F1-optimal | III | 0.52 | 0.795 | 0.842 | 0.537 | 0.636 | 0.700 | 0.512 |
| balanced (Youden) | III | 0.66 | 0.773 | 0.708 | **0.244** | **0.364** | **0.400** | **0.207** |
| default 0.5 | II | 0.50 | 0.645 | 0.627 | 0.283 | 0.209 | 0.500 | 0.312 |
| F1-optimal | II | 0.33 | 0.680 | 0.864 | 0.604 | 0.547 | 0.750 | 0.618 |

Mean F1 over phases: default 0.737, F1-optimal **0.752**, Youden 0.737.
**Insight:** F1 (positive class = success) rewards predicting the majority class, so the F1-optimal operating point *doubles* the share of Phase II failures predicted successful.
Whether a model "catches failing trials" is a separate question from F1, and it is answered by the operating point.

Recall on successful trials, original → improved: I 0.81 → 0.81, II 0.59 → 0.63, III 0.86 → 0.85.
Fewer false positives were therefore *not* bought with more false negatives.
Files: `outputs/p2_error_by_reason_comparison.csv` (includes Wilson CIs and true/predicted label counts) and figure `p2_fig5`.

## Insights worth writing up (ranked)
1. **Pooling phases is the cheapest big win, and it is a data-size story.** Phase I has only 1,012 training rows. One pooled LR with a phase one-hot raises CV AUC from 0.709 to 0.727.
   In the final model, Phase I gains the most: AUC +0.031, and the error on failures drops from 53% to 39%, with Recruitment failures falling from 55% to 27%.
   Eligibility patterns that predict recruitment failure are learned from the larger Phase II set and transferred.
2. **A drug's or disease's track record is the most useful new feature, and design features only help in combination.** Removing `hist` costs 0.033 CV AUC.
   `design` alone is weak (0.656), but `design + hist` beats `hist` alone (0.704 vs 0.696).
   In the ablation-by-reason table (notebook Section 6), the full dense model has the lowest Efficacy (0.42) and Safety (0.41) error of the single models. Drug and disease history is what speaks to efficacy and safety.
3. **Why including everything fails (mirrors Part 1).** Adding 1024 fingerprint bits, 1,195 pathway indicators and the action multi-hot to the pooled LR drops CV AUC from 0.726 to 0.708.
   These blocks are sparse, 54% missing (no matched drug), and redundant with `hist`/`tani`, which already summarise the drug in one outcome-aware number. On ~8k rows the extra dimensions mostly add variance.
   Fingerprints and pathways are neutral inside HGB (trees ignore useless splits); Tanimoto and descriptors are ±0 within CV noise.
4. **Phase III failures stay hard, and that makes sense.** Phase III trials have already passed Phase II, so failures are mostly efficacy or recruitment surprises that nothing in the registration text predicts.
   Its Efficacy, Other and Recruitment error rates did not improve (n = 10, 10, 22; the 95% CIs span about ±0.3).
   Changes of one or two trials in the Funding, Efficacy and Other cells are noise; only the "All failures" and the large Recruitment / No-reason cells support conclusions.
5. **Category sizes and label noise.** 86% of reason labels on the val failures are Part 1 *predictions*. Part 1 val per-reason F1 was 0.98 for Recruitment but only 0.57–0.67 for Funding and Other, so the Funding and Other rows are both tiny and noisy.
   For the original model, error rates on true-label vs predicted-label trials differ (e.g. Recruitment 0.21 true vs 0.39 predicted), but those true-label groups are only n = 2–19.
6. **Naive Bayes ranks but cannot give probabilities.** Its AUC is 0.69, yet its log loss is 3.6, 6× worse than LR, because the independence assumption over thousands of tokens pushes posteriors to 0 or 1 (`p2_fig2`).
   The stacker gives it weight 0.00. LR is already well calibrated, so Platt scaling alone changes it by only 0.002 log loss.
7. **The success rate rises sharply with start year, and test is entirely out of range.** Phase III goes from 48% (≤2004) to 89% (2012–13).
   Trees cannot extrapolate: on the temporal holdout, HGB with a year feature gives worse log loss (0.588) than the stack without year (0.559).
   The stacker's per-phase logistic year trend gets the base rates closest, with predicted vs actual 2010–13 means of 0.68/0.59/0.77 vs 0.65/0.57/0.77.
   **Caveat:** all strategies still under-predict the steep 2012–13 rise in Phase III (`p2_fig4`).
8. **Expected val → test gap.** On the temporal holdout, AUC is 0.715 for the stack vs 0.760 on val, and the original LR drops 0.736 → 0.680. The stack degrades less than the original.
   The test gap should be larger again: test starts 1–8 years after training ends, the test is 1/3 Phase I by construction (vs 13% of train), and its criteria are longer, with more exclusion criteria (13.1 vs 9.2 bullets).
9. **Adversarial validation, and an artefact we caught.**
   - A classifier separates train+val from test with AUC 0.78 on design/drug features and 0.86 on TF-IDF text, so there is real covariate shift.
     Top drifting features: criteria length, the Tanimoto top-similarity (0.42 → 0.31: test drugs are less similar to any training drug), the Phase I share, the number of exclusion criteria and lab thresholds.
   - With the history encodings included, the AUC jumps to 0.97. A **same-era control** (train vs val, history encodings only) is just as separable (0.97).
     The cause is that out-of-fold encodings (80% of data) and full-fit encodings take slightly different exact values for the same drug, and a tree memorises them. That is an encoding artefact, not drift.

10. **The improvement is statistically supported for ranking, calibration and F1, but not for PR-AUC.** The stack wins in 5/5 CV folds, and the bootstrap CIs exclude 0 for ROC-AUC, log loss and F1.
    Phase III cannot be ranked better with these inputs: prior-phase history and class balancing both fail.
    Its failure detection is a threshold choice: a balanced threshold cuts Phase III's failure error from 56% to 24% at the cost of recall on successes (0.85 → 0.71).

## Figures (`midterm/figures/`); suggest 3 for the report
- `p2_fig5_error_by_reason.png` (must-have; per-phase error by reason, original vs improved, with CIs)
- `p2_fig6_threshold_tradeoff.png` (ROC per phase with operating points; the trade-off figure)
- `p2_fig4_temporal_holdout.png` (time-shift / year-extrapolation story)
- `p2_fig3_feature_ablation.png` (val + CV AUC for every feature block)
- `p2_fig1_success_by_year.png`, `p2_fig2_calibration_baselines.png` (optional)

## Kaggle
- **Test log loss 0.59274 (9th on the leaderboard at submission time, 2026-11-06).** Val log loss was 0.569, so the val → test gap is +0.024.
  The temporal holdout (fit ≤ 2009, eval 2010–13) predicted degradation of this kind, and the gap is moderate given test starts 1–8 years after training ends.
- Primary: `outputs/part2_submission.csv` (stack + year trend, capped at 2013). Mean predicted P(success) on test: I 0.61, II 0.55, III 0.76, vs train+val rates of 0.57/0.48/0.65.
- Alternate: `outputs/part2_submission_alt_noyear.csv` (same stack without the year trend): **test log loss 0.59799**.
  The year trend therefore helps on test by 0.0053 log loss, more than on the temporal holdout (0.001).
  Because the year is capped at 2013, every test trial is anchored to the most recent (2013) base rate instead of the all-years average.
  Base rates keep rising after the training window, and that anchoring matters more the further out the test years are.
  The primary submission was chosen on the holdout *before* seeing either test score; the alternate was submitted only as a diagnostic.

## References (Part 2 additions)
Fu et al. 2022 (HINT; historical drug/disease outcome features) · Wolpert 1992 (stacked generalization) · Platt 1999 (probability calibration) ·
Ke et al. 2017 / scikit-learn HistGradientBoosting · Rogers & Hahn 2010 (Morgan/ECFP) · Bajusz et al. 2015 (Tanimoto similarity) ·
Wilson 1927 (score interval) · Sparck Jones 1972 (TF-IDF) · Pedregosa et al. 2011 (scikit-learn) · RDKit · Reactome (Gillespie et al. 2022).
