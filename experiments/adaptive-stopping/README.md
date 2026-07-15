# Adaptive stopping counterfactual replay

**Recommendation: revise the targeted strategy and keep it experimental.** Do not change the
balanced default yet, and do not proceed directly to a live trial from this one replay.

The opt-in targeted allocator delivers the expected call reduction, but not by stopping early:
it uses all 10 five-sample rounds while selecting fewer pairs per round. Against the full stored
board, plain targeted allocation saves **79.7%** of comparisons, while the 3× size rule saves
**82.9%**. Both preserve top-3 membership, but both swap ranks 2 and 3 and move individual ELOs
substantially. The size rule adds only 3.1 percentage points of savings over plain targeted while
increasing median/max ELO drift.

No production bug was found. This is therefore an experiment/documentation-only change.

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

All three adaptive runs still exhaust the sample batches. See
[`results-sentinels-excluded.json`](results-sentinels-excluded.json) for full details.

## Determinism

Each of the four primary strategies was replayed twice. Selected comparison keys, interim rank
orders, ELOs, bootstrap CIs, adjacent-pair decisions, final annotations, stop round, and stop
reason matched exactly. This confirms deterministic execution in the tested environment, helped
by the fixed bootstrap seed and deterministic equal-ELO tie-break.

It does not establish reproducibility across different SciPy/NumPy versions or statistical
validity under repeated data collection.

## Limitations

1. **Counterfactual, not independent.** The selected outcomes are a subset of the same run used as
   the reference. This does not test a new judge run, judge drift, API nondeterminism, or a new
   sample of pages.
2. **Outcome-conditioned sampling.** Later targeted pairs depend on interim outcomes and ranks.
   Ordinary percentile bootstrap CIs computed after selection do not account for that adaptive,
   optional-stopping process and may be optimistic.
3. **The bootstrap does not replay the policy.** Production resamples selected comparisons as if
   the selected set were fixed. It does not re-run allocation inside each bootstrap replicate.
4. **Comparison-level resampling ignores page clustering.** Many pair outcomes share one page and
   are correlated. A page/round-clustered bootstrap would better match the sampling unit.
5. **Bradley–Terry fit under sparse, nonuniform allocation.** The graph is connected, so a fit is
   identifiable, but adjacency-driven edge counts can amplify model misspecification and shift the
   global ELO scale relative to a balanced grid.
6. **One collection and one stored judge run.** Britannica cannot establish general behavior for
   manuscripts, tables, noisy scans, or other judges.
7. **Unobserved failed judge calls.** Six attempted comparisons have no published valid verdict.
   The replay cannot know how a different allocation would have distributed those failures, so
   savings are relative to 4,293 valid stored outcomes rather than the 4,299 attempted calls.
8. **Historical sentinel rows.** The primary board predates the current exclusion policy; the
   robustness replay is closer to current input integrity semantics but has only 13 rankable
   models.

## Recommendation and next experiment

**Revise strategy; keep `targeted` opt-in and `balanced` as the default.** Specifically, evaluate a
revision that:

1. uses a per-pair balanced warm-up floor rather than only the aggregate 273-outcome threshold;
2. injects periodic balanced/exploration batches so early adjacency does not permanently starve
   other edges;
3. monitors rank/top-k stability and ELO drift across rounds, not only overlapping adjacent
   marginal CIs; and
4. evaluates page-clustered or policy-aware uncertainty for adaptive runs.

Replay that revision on several existing result boards before one explicitly budgeted live trial.
The 3× practical rule can remain as an annotation/stopping preference, but this board does not
support making it a default or treating it as statistical resolution.

## Reproduce

From the repository root:

```bash
uv sync --dev
uv run python experiments/adaptive-stopping/replay.py
```

The full run takes about five minutes on the machine used for this report. The pinned dataset
revision is downloaded read-only. The script contains no judge backend construction and no Hub
push/upload call.

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
- [`results-sentinels-excluded.json`](results-sentinels-excluded.json): robustness replay

Validation commands:

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ experiments/adaptive-stopping/replay.py
uv run ty check src/ experiments/adaptive-stopping/replay.py
```
