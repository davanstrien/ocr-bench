# Adaptive stopping counterfactual replay

**Recommendation: keep targeted allocation experimental and do not run a live trial.** Follow-up
replays tested stronger adaptive allocation, fixed outcome-independent designs, and seven
published boards spanning five collections and three Rubenstein judges. A newly collected
independent 14-model MOH board confirms there is no basis for a production change or live targeted
trial.

The opt-in targeted allocator delivers the expected call reduction, but not by stopping early:
it uses all 10 five-sample rounds while selecting fewer pairs per round. Against the full stored
board, plain targeted allocation saves **79.7%** of comparisons, while the 3× size rule saves
**82.9%**. Both preserve top-3 membership, but both swap ranks 2 and 3 and move individual ELOs
substantially. The size rule adds only 3.1 percentage points of savings over plain targeted while
increasing median/max ELO drift.

No production bug was found. This remains an experiment/documentation-only change.

## Scope

This experiment follows [issue #35](https://github.com/davanstrien/ocr-bench/issues/35) and
[PR #53](https://github.com/davanstrien/ocr-bench/pull/53), using main at `ec802fa`.

Data is pinned to:

- repo: `davanstrien/ocr-bench-britannica-results`
- config: `comparisons`
- revision: `48a0f42de26009892d2784a3a97d6d61525f4040`
- 4,293 stored outcomes, 14 models, 50 samples, and all 91 possible model pairs

The source metadata records 4,299 attempted comparisons and 4,293 valid comparisons. Only the
4,293 published verdicts are observable, so “used” and “saved” below count stored valid outcomes,
not reconstructed API attempts.

This is a **counterfactual replay**, not independent validation. It asks which already-stored
outcomes each policy would have selected. It makes no judge API calls and writes nothing to the
Hub.

## Method

The replay:

1. Processes sample indices in production batches of five: 0–4, 5–9, ..., 45–49.
2. Starts targeted runs with balanced evidence.
3. First checks after production's `max(3 * n_pairs, 20)` threshold: 273 outcomes here.
4. Uses production `compute_elo` with 1,000 bootstrap replicates and seed 42.
5. Uses production `classify_adjacent_pairs`, `unresolved_pairs`, pair counting, model-size
   parsing, and practical-preference helpers.
6. Recomputes adjacency after every batch, just as the CLI does.
7. Separately evaluates fixed pair-balanced and mixed-random designs at budgets 700, 1,200, and
   2,000 over allocation seeds 42–46; these selectors never inspect winner values.

The primary replay intentionally treats all 4,293 published rows as outcomes because that is the
requested reference board and reproduces the historical leaderboard. The data includes 628 rows
containing OCR error sentinels, judged before the current sentinel exclusion policy. A separate
robustness replay applies the current exclusion rule below.

### Metrics

- Rank correlations compare complete 14-model orders with the full stored board.
- ELO deltas use the same 14-model Bradley–Terry centering.
- Pair coverage is observed undirected edges out of 91; min/median/max are direct outcomes per
  observed edge.
- `stat/practical/unresolved` counts the final 13 adjacent pairs. “Practical” means the opt-in
  smaller-model deployment preference, not statistical resolution.

## Primary results

All adaptive policies reached round 10 and exhausted the available sample batches. **None met its
stopping criteria early.**

| Strategy | Used | Saved | Stop | Kendall τ | Spearman ρ | Top-3 members | Top-3 order | Median abs(ΔELO) | Max abs(ΔELO) | Pair coverage (min/med/max) | Stat/practical/unresolved |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|---|
| Full stored board | 4,293 | 0.0% | reference, round 10 | 1.000 | 1.000 | 3/3 | yes | 0.0 | 0.0 | 91/91 (43/46/50) | 4/0/9 |
| Balanced adaptive | 4,293 | 0.0% | samples exhausted, round 10 | 1.000 | 1.000 | 3/3 | yes | 0.0 | 0.0 | 91/91 (43/46/50) | 4/0/9 |
| Targeted | 870 | 79.7% | samples exhausted, round 10 | 0.934 | 0.982 | 3/3 | **no** | 36.3 | 85.5 | 91/91 (2/5/46) | 1/0/12 |
| Targeted + 3×, min 10 | 736 | 82.9% | samples exhausted, round 10 | 0.912 | 0.978 | 3/3 | **no** | 59.0 | 105.9 | 91/91 (2/5/41) | 1/6/6 |

The rough 870 targeted estimate is reproduced exactly. The rigorous current-helper result for the
3× rule is 736 rather than the rough ~715.

### Final rank order

| Rank | Full / balanced | Targeted | Targeted + 3× |
|---:|---|---|---|
| 1 | dots.mocr | dots.mocr | dots.mocr |
| 2 | LightOnOCR-2-1B | GLM-OCR | GLM-OCR |
| 3 | GLM-OCR | LightOnOCR-2-1B | LightOnOCR-2-1B |
| 4 | olmOCR-2-7B-1025-FP8 | olmOCR-2-7B-1025-FP8 | olmOCR-2-7B-1025-FP8 |
| 5 | NuExtract3 | FireRed-OCR | FireRed-OCR |
| 6 | Qianfan-OCR | NuExtract3 | NuExtract3 |
| 7 | FireRed-OCR | Qianfan-OCR | Qianfan-OCR |
| 8 | Unlimited-OCR | Unlimited-OCR | PaddleOCR-VL-1.6 |
| 9 | PaddleOCR-VL-1.6 | PaddleOCR-VL-1.6 | Unlimited-OCR |
| 10 | DeepSeek-OCR | DeepSeek-OCR | DeepSeek-OCR |
| 11 | PP-OCRv6_medium | PP-OCRv6_medium | PP-OCRv6_medium |
| 12 | DeepSeek-OCR-2 | DeepSeek-OCR-2 | DeepSeek-OCR-2 |
| 13 | tesseract-5 | tesseract-5 | tesseract-5 |
| 14 | dots.ocr | dots.ocr | dots.ocr |

Top-3 **membership** is stable, but its ordering is not: GLM-OCR and LightOnOCR-2-1B swap.
Targeting also changes ranks 5–7; the size rule additionally swaps ranks 8–9.

### Per-model ELO change

Balanced adaptive consumes the complete board, so every balanced delta is zero. Parentheses show
change from the full stored board.

| Model | Full ELO | Targeted ELO (Δ) | Targeted + 3× ELO (Δ) |
|---|---:|---:|---:|
| rednote-hilab/dots.mocr | 1745.4 | 1792.9 (+47.5) | 1810.3 (+64.9) |
| lightonai/LightOnOCR-2-1B | 1741.3 | 1749.4 (+8.1) | 1737.3 (-3.9) |
| zai-org/GLM-OCR | 1738.2 | 1771.9 (+33.8) | 1773.3 (+35.2) |
| allenai/olmOCR-2-7B-1025-FP8 | 1719.2 | 1642.8 (-76.4) | 1635.9 (-83.3) |
| numind/NuExtract3 | 1689.6 | 1605.7 (-83.9) | 1585.8 (-103.8) |
| baidu/Qianfan-OCR | 1568.1 | 1539.9 (-28.2) | 1530.4 (-37.7) |
| FireRedTeam/FireRed-OCR | 1557.4 | 1608.7 (+51.2) | 1624.9 (+67.5) |
| baidu/Unlimited-OCR | 1545.2 | 1506.3 (-38.9) | 1480.1 (-65.1) |
| PaddlePaddle/PaddleOCR-VL-1.6 | 1463.0 | 1454.6 (-8.4) | 1491.0 (+28.0) |
| deepseek-ai/DeepSeek-OCR | 1452.0 | 1426.5 (-25.4) | 1398.9 (-53.1) |
| PaddlePaddle/PP-OCRv6_medium | 1397.1 | 1377.7 (-19.4) | 1376.9 (-20.1) |
| deepseek-ai/DeepSeek-OCR-2 | 1380.2 | 1377.3 (-2.9) | 1357.2 (-23.0) |
| tesseract-5 | 1112.6 | 1198.1 (+85.5) | 1218.5 (+105.9) |
| rednote-hilab/dots.ocr | 890.7 | 948.2 (+57.5) | 979.3 (+88.7) |

The targeted graph remains connected and retains direct evidence for every pair because of the
balanced warm-up. Connectivity alone is not enough: targeted median pair evidence falls from 46
to 5, with some pairs frozen at 2 outcomes. The nonuniform edge weights are reflected in the ELO
movement.

## Statistical resolution versus practical preference

At the end of the full board, only 4 of 13 adjacent pairs are statistically resolved; 9 still have
overlapping marginal CIs. Plain targeted ends with only 1 resolved and 12 unresolved.

With the 3× rule, the final board has:

- 1 statistically resolved adjacent pair;
- 6 overlapping pairs annotated as practical smaller-model preferences; and
- 6 still unresolved pairs.

The practical annotations produced are:

| Prefer smaller | Compared with | Parameter ratio |
|---|---|---:|
| FireRedTeam/FireRed-OCR | allenai/olmOCR-2-7B-1025-FP8 | 4.0× |
| PaddlePaddle/PP-OCRv6_medium | deepseek-ai/DeepSeek-OCR | 115.9× |
| PaddlePaddle/PP-OCRv6_medium | deepseek-ai/DeepSeek-OCR-2 | 98.6× |
| PaddlePaddle/PaddleOCR-VL-1.6 | baidu/Qianfan-OCR | 5.2× |
| PaddlePaddle/PaddleOCR-VL-1.6 | baidu/Unlimited-OCR | 3.7× |
| zai-org/GLM-OCR | rednote-hilab/dots.mocr | 3.3× |

These do **not** establish equivalence or alter ELO/rank. CI overlap is a failure to statistically
separate two marginal estimates, not evidence that model quality is equal. The annotation says
only: given overlap, enough direct pair evidence, and the configured parameter proxy, prefer the
smaller deployment. Parameter count is also not a direct latency, memory, price, or throughput
measurement.

## Sensitivity

All sensitivity runs also exhausted round 10 rather than meeting all stopping criteria.

| Size ratio | Min evidence | Used | Saved | Kendall τ | Spearman ρ | Top-3 members | Top-3 order | Median abs(ΔELO) | Max abs(ΔELO) | Unresolved |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2× | 10 | 629 | 85.3% | 0.890 | 0.952 | **2/3** | no | 79.2 | 167.4 | 3 |
| 3× | 5 | 735 | 82.9% | 0.824 | 0.912 | 3/3 | no | 79.0 | 155.3 | 8 |
| 3× | 10 | 736 | 82.9% | 0.912 | 0.978 | 3/3 | no | 59.0 | 105.9 | 6 |
| 3× | 15 | 786 | 81.7% | 0.934 | 0.982 | 3/3 | no | 46.0 | 98.4 | 7 |
| 5× | 10 | 810 | 81.1% | 0.934 | 0.982 | 3/3 | no | 35.8 | 85.1 | 9 |

The response is not monotonic because each interim board changes adjacency and therefore later
allocation. A permissive 2× threshold loses one full-board top-3 member. Lowering minimum evidence
from 10 to 5 barely saves another comparison but materially worsens rank correlation and ELO
error. A 15-sample minimum or 5× ratio is less aggressive and tracks the full board better, but
neither restores top-3 ordering or achieves early stopping.

## Follow-up targeted-v2 replay

A second replay tested the proposed next step without changing production code. The gate was
fixed before running:

- exact full-board top-3 order;
- Kendall τ ≥ 0.95;
- maximum absolute ELO delta ≤ 50; and
- at least 60% of stored outcomes saved.

“Explore every 3” means every third post-warm-up allocation batch is balanced; the other two are
targeted. “Annotate 3×” computes practical smaller-model annotations on the final board but does
not let that rule remove pairs from sampling. “Size-stop 3×” does let it control sampling, matching
the production option.

| Follow-up strategy | Used | Saved | Kendall τ | Spearman ρ | Exact top-3 order | Median abs(ΔELO) | Max abs(ΔELO) | Min pair evidence | Gate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| Per-pair warm-up 5 | 1,153 | 73.1% | 0.912 | 0.969 | yes | 74.0 | 157.5 | 5 | **fail** |
| Per-pair warm-up 10 | 1,561 | 63.6% | 0.934 | 0.982 | yes | 50.0 | 99.1 | 10 | **fail** |
| Balanced exploration every 3 | 2,023 | 52.9% | 0.956 | 0.991 | no | 31.8 | 98.6 | 17 | **fail** |
| Warm-up 5 + explore 3 + annotate 3× | 1,946 | 54.7% | 0.934 | 0.987 | no | 29.8 | 88.2 | 15 | **fail** |
| Warm-up 10 + explore 3 + annotate 3× | 2,313 | 46.1% | 0.956 | 0.991 | yes | 37.1 | 99.6 | 18 | **fail** |
| Warm-up 5 + explore 3 + size-stop 3× | 1,820 | 57.6% | 0.934 | 0.987 | no | 31.8 | 90.8 | 15 | **fail** |

The candidate suggested after the first replay—warm-up 5 plus every-third-batch exploration—did
not pass. Periodic balanced batches improved global rank correlation and reduced median ELO drift,
but consumed enough outcomes to miss the savings gate and still left a large worst-model delta.
Warm-up floors alone restored the full-board top-3 order, but did not stabilize the rest of the
board. No follow-up strategy met its stopping criteria before round 10.

Separating annotation from allocation was directionally preferable: with warm-up 5 and exploration,
annotation-only used 1,946 outcomes and had max drift 88.2, while size-controlled stopping used
1,820 and had max drift 90.8. The additional 126-outcome saving is not enough evidence to let a
practical deployment preference influence statistical allocation.

The exact order of the full board's top three is itself weakly identified—their CIs overlap—so it
should not become a universal product criterion. It remains useful here as a deliberately strict
counterfactual fidelity check, alongside full-order correlation and ELO drift.

## Fixed outcome-independent designs

The next proposed direction was also replayed. Two fixed designs were evaluated at predeclared
budgets of 700, 1,200, and 2,000 outcomes over seeds 42–46:

- **Pair-balanced:** shuffle outcomes within each model pair, then assign equal per-pair quotas.
- **Mixed-random:** retain the first five-page balanced warm-up, then fill the budget by seeded
  pair-balanced exploration independent of interim winners.

Both condition only on which valid stored rows exist; neither reads the winner when allocating.
The table reports medians and ranges across five seeds. Top-3 columns are counts out of five.

| Design | Budget | Saved | Kendall τ median [range] | Top-3 set/order | Median abs(ΔELO), median | Max abs(ΔELO), median/worst | Min pair evidence | Gate passes |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Pair-balanced | 700 | 83.7% | 0.868 [0.758–0.978] | 1/1 | 34.9 | 75.2 / 116.4 | 7 | 0/5 |
| Pair-balanced | 1,200 | 72.0% | 0.890 [0.780–0.956] | 2/0 | 18.8 | 49.1 / 89.0 | 13 | 0/5 |
| Pair-balanced | 2,000 | 53.4% | 0.868 [0.846–0.956] | 1/1 | 17.6 | 37.7 / 61.2 | 21 | 0/5 |
| Mixed-random | 700 | 83.7% | 0.868 [0.868–0.890] | 5/1 | 48.2 | 146.7 / 191.3 | 7 | 0/5 |
| Mixed-random | 1,200 | 72.0% | 0.934 [0.934–0.956] | 5/1 | 28.7 | 79.8 / 122.8 | 13 | 0/5 |
| Mixed-random | 2,000 | 53.4% | 0.956 [0.934–0.978] | 5/2 | 13.7 | 46.3 / 61.4 | 21 | 0/5 |

No fixed design passed the original gate. Mixed-random preserved top-3 **membership** in all 15
runs and reached median Kendall τ 0.956 at budget 2,000, but that budget saved only 53.4%, exact
top-3 order held for 2/5 seeds, and worst-seed max ELO drift remained 61.4. At budget 1,200 it
saved 72.0%, but median max drift was 79.8.

The seed ranges are as important as the medians: a single favorable seed would have overstated
fidelity. Outcome-independent allocation removes the targeted policy's optional-sampling feedback,
but does not make a small board equivalent to the full one. The fixed designs also show that
pair-count balance alone is insufficient; preserving a common balanced warm-up made top-3
membership much more stable than independently sampling each pair.

## Cross-collection, page-clustered replay

The fixed designs were then replayed on seven suitable published boards without new judge calls.
Revisions are pinned in [`multi_board.py`](multi_board.py).

| Board | Collection/role | Outcomes | Models | Samples |
|---|---|---:|---:|---:|
| Rubenstein 30B | independent manuscript-card collection | 299 | 4 | 50 |
| Rubenstein jury | same collection, judge sensitivity | 300 | 4 | 50 |
| Rubenstein Kimi | same collection, judge sensitivity | 300 | 4 | 50 |
| UFO 30B | independent collection | 294 | 4 | 49 |
| BPL | independent, incomplete board | 147 | 4 | 41 |
| Britannica Qwen35 | same corpus, earlier 6-model/judge sensitivity | 720 | 6 | 50 |
| MOH table fidelity | independent large-grid collection | 4,505 | 14 | 50 |

Rubenstein's three judges are not counted as three independent collections. This gives five
collection groups and two independent 14-model boards: Britannica and MOH.

### Uncertainty method

For each board, design, budget fraction (25%, 40%, 60%), and allocation seed (42–46), the replay:

1. resamples complete `sample_idx` page clusters 200 times;
2. refits the full board inside each page bootstrap;
3. reruns the fixed allocation inside that same replicate; and
4. compares the selected board with its paired bootstrap full board.

This is 1,000 design-aware page-bootstrap evaluations per board/design/budget. A full-board pair is
called robust only when its direction repeats in at least 95% of full-board page bootstraps. The
cross-board criteria were predeclared as:

- clustered top-3 membership agreement ≥90%;
- agreement on robust full-board pairs ≥95%; and
- median clustered Spearman ρ ≥0.90.

Exact ordering among unresolved models and absolute ELO drift are still reported, but are not used
as pass criteria after the first replay showed that they can be unstable even for the full board.

### Full-board stability ceiling

| Board | Robust pairs | Possible | Full-board top-3 set stability | Full-board top-3 order stability |
|---|---:|---:|---:|---:|
| Rubenstein 30B | 4 | 6 | 100.0% | 65.5% |
| Rubenstein jury | 3 | 6 | 83.0% | 46.5% |
| Rubenstein Kimi | 5 | 6 | 99.5% | 89.0% |
| UFO 30B | 5 | 6 | 100.0% | 84.0% |
| BPL | 4 | 6 | 52.0% | 28.5% |
| Britannica Qwen35 | 13 | 15 | 100.0% | 65.0% |
| MOH table fidelity | 51 | 91 | 19.0% | 10.5% |

BPL, the Rubenstein jury board, and MOH cannot meet a 90% top-3 stability target reliably even when
all stored outcomes are used. Their failures below are therefore evidence that the reference board
is under-resolved, not simply that subsampling is bad. MOH still has 51 robust pair directions;
its instability is concentrated around exact ordering rather than every relationship.

### Results at a 40% outcome budget

| Board | Pair-balanced | Mixed-random | Clustered top-3 rate (pair/mixed) | Robust-pair agreement (pair/mixed) |
|---|---|---|---:|---:|
| Rubenstein 30B | pass | pass | 98.4% / 98.6% | 97.6% / 97.7% |
| Rubenstein jury | fail | fail | 79.4% / 77.2% | 97.8% / 97.4% |
| Rubenstein Kimi | pass | pass | 97.6% / 98.0% | 97.5% / 97.4% |
| UFO 30B | pass | pass | 99.9% / 99.6% | 100.0% / 99.9% |
| BPL | fail | fail | 65.4% / 62.2% | 86.9% / 85.9% |
| Britannica Qwen35 | pass | pass | 98.8% / 98.9% | 98.3% / 98.3% |
| MOH table fidelity | fail | fail | 47.4% / 37.2% | 98.4% / 98.3% |

At 40%, both fixed designs pass on Rubenstein 30B, Rubenstein Kimi, UFO, and the 6-model
Britannica board; both fail on the intrinsically unstable Rubenstein jury, BPL, and MOH boards.
At 25%, results are more collection-dependent. Increasing to 60% still cannot rescue unstable
reference boards.

There is no consistent winner between pair-balanced and mixed-random allocation on these small
model grids. Board/judge stability matters more than the choice between the two fixed selectors.
Small samples also produce heavy ELO tails—Britannica Qwen35 has extreme page-bootstrap ELO
outliers despite stable rank decisions—so rank/pairwise summaries are safer than treating ELO
error as approximately Gaussian.

MOH supplies the previously missing independent large grid. Its plain targeted replay uses 996
outcomes (77.9% saved) but moves the full order to Kendall τ 0.846 and swaps ranks 2–3; size-aware
targeting uses 803 outcomes and falls to τ 0.780. Neither stops early. Full MOH details are in
[`moh/README.md`](moh/README.md).

Across the two 14-model boards, targeted allocation consistently saves roughly 80% while moving
rankings and ELOs materially. See [`multi-board-summary.csv`](multi-board-summary.csv) and
[`multi-board-results.json`](multi-board-results.json).

## Current sentinel-policy robustness check

The primary result must use the requested 4,293-row stored board. However, current production
would exclude the 628 historical comparisons containing an OCR error sentinel and mark dots.ocr
as a fully failed model. Replaying the remaining 3,665 rows over 13 models gives the same broad
conclusion:

| Strategy | Used | Saved vs 3,665 | Kendall τ | Spearman ρ | Top-3 members/order | Median/max abs(ΔELO) |
|---|---:|---:|---:|---:|---|---:|
| Balanced | 3,665 | 0.0% | 1.000 | 1.000 | 3/3, yes | 0.0 / 0.0 |
| Targeted | 753 | 79.5% | 0.923 | 0.984 | 3/3, no | 27.0 / 69.0 |
| Targeted + 3×, min 10 | 624 | 83.0% | 0.897 | 0.973 | 3/3, no | 48.6 / 86.0 |

All three adaptive runs still exhaust the sample batches. The targeted-v2 variants also fail the
predeclared gate under current sentinel handling. The closest on ELO drift—warm-up 5 plus
exploration 3, annotation-only—uses 1,684 outcomes (54.1% saved), has median/max absolute ELO
deltas of 18.1/57.0, and still swaps the statistically unresolved models in ranks 2 and 3.
For the fixed designs under current sentinel handling, mixed-random at budget 1,200 passed the
gate for 2/5 seeds but not robustly; no design/budget passed for every seed. Mixed-random at 2,000
had median/worst max ELO drift of 22.3/32.3, but saved only 45.4% and reproduced exact top-3 order
in 0/5 seeds. See [`results-sentinels-excluded.json`](results-sentinels-excluded.json) for full
details.

## Determinism

Each of the four primary strategies and the warm-up-5/exploration-3 annotation-only candidate was
replayed twice. Selected comparison keys, interim rank orders, ELOs, bootstrap CIs, adjacent-pair
decisions, final annotations, stop round, and stop reason matched exactly. This confirms
deterministic execution in the tested environment, helped by the fixed bootstrap seed and
deterministic equal-ELO tie-break.

The fixed selectors are also deterministic for a fixed seed, with unit tests covering exact
budgets, pair balance, warm-up preservation, and repeat selection. Their five-seed spread measures
allocation sensitivity, not repeated-data statistical uncertainty. None of this establishes
reproducibility across different SciPy/NumPy versions or statistical validity under repeated data
collection.

## Limitations

1. **Counterfactual, not independent.** The selected outcomes are a subset of the same run used as
   the reference. This does not test a new judge run, judge drift, API nondeterminism, or a new
   sample of pages.
2. **Outcome-conditioned sampling.** Later targeted pairs depend on interim outcomes and ranks.
   Ordinary percentile bootstrap CIs computed after selection do not account for that adaptive,
   optional-stopping process and may be optimistic.
3. **Production uncertainty does not replay the policy.** Production resamples selected
   comparisons as if the selected set were fixed. The cross-board experiment reruns fixed
   allocation inside page bootstraps, but this is not yet a production CI method for targeted
   optional stopping.
4. **Production comparison-level resampling ignores page clustering.** Many pair outcomes share
   one page and are correlated. The cross-board follow-up addresses this experimentally; ordinary
   published CIs still do not.
5. **Bradley–Terry fit under sparse, nonuniform allocation.** The graph is connected, so a fit is
   identifiable, but adjacency-driven edge counts can amplify model misspecification and shift the
   global ELO scale relative to a balanced grid.
6. **Large-grid evidence is still limited.** Britannica and MOH provide two independent 14-model
   boards under different criteria, and both warn against targeted allocation. More collections
   would be needed before claiming a universal effect size.
7. **Unobserved failed judge calls.** Six attempted comparisons have no published valid verdict.
   The replay cannot know how a different allocation would have distributed those failures, so
   savings are relative to 4,293 valid stored outcomes rather than the 4,299 attempted calls.
8. **Historical sentinel rows.** The primary board predates the current exclusion policy; the
   robustness replay is closer to current input integrity semantics but has only 13 rankable
   models.
9. **Legacy cross-board schemas.** Several older boards do not store OCR text, so historical
   sentinel rows cannot be rechecked. Five additional boards have only four models, limiting their
   contribution to sparse large-grid allocation evidence.
10. **Judge variants are not independent data.** The three Rubenstein boards measure sensitivity
    to judge choice on the same pages; they do not triple the collection-level evidence.

## Recommendation and next experiment

**Keep `targeted` opt-in, retain `balanced` as the default, and do not implement the tested v2 or
fixed designs in production.** MOH provides the missing independent 14-model test and fails the
fidelity criteria: targeted allocation does not preserve the full board closely, while the full
board's own leading order is highly page-sensitive.

The evidence is now sufficient to stop allocation tuning and reject a live targeted trial for this
strategy. The next product-level question is different: whether leaderboards should report
page-clustered rank tiers or robust pairwise relations rather than a single exact order. That work
requires a separate design issue and must not be smuggled into this experiment PR.

Keep the 3× rule as a post-hoc deployment annotation only. Current evidence does not support
letting it influence sampling, making it a default, or treating it as statistical resolution.

## Reproduce

From the repository root:

```bash
uv sync --dev
uv run python experiments/adaptive-stopping/replay.py
```

The full primary, sensitivity, targeted-v2, and fixed-design run takes about eleven minutes on the
machine used for this report. The pinned dataset
revision is downloaded read-only. The script contains no judge backend construction and no Hub
push/upload call.

Cross-collection page-clustered replay:

```bash
uv run python experiments/adaptive-stopping/multi_board.py
```

Optional current-sentinel-policy replay:

```bash
uv run python experiments/adaptive-stopping/replay.py \
  --exclude-sentinel-comparisons \
  --skip-sensitivity \
  --repeats 1
```

Generated artifacts:

- [`results.json`](results.json): complete metrics, ranks, CIs, annotations, and round history
- [`strategy-summary.csv`](strategy-summary.csv): strategy-level metrics
- [`elo-deltas.csv`](elo-deltas.csv): per-model ELO and deltas
- [`round-history.csv`](round-history.csv): batch-by-batch allocation and decisions
- [`static-design-summary.csv`](static-design-summary.csv): five-seed fixed-design aggregates
- [`multi-board-summary.csv`](multi-board-summary.csv): cross-board design-aware aggregates
- [`multi-board-results.json`](multi-board-results.json): pinned board and page-bootstrap details
- [`moh/README.md`](moh/README.md): independent 14-model replay conclusions
- [`moh/results.json`](moh/results.json): complete pinned MOH replay metrics
- [`results-sentinels-excluded.json`](results-sentinels-excluded.json): robustness replay

Validation commands:

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ experiments/adaptive-stopping/*.py
uv run ty check src/ experiments/adaptive-stopping/*.py
```
