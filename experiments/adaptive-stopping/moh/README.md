# MOH independent 14-model replay

This follow-up applies the adaptive-stopping experiment to a second independent 14-model board:
`davanstrien/ocr-bench-moh-results` at revision
`05d24695f131432c38984dd2cb55f5a18a3c2580`.

The reference was collected non-adaptively over 50 Medical Officer of Health report pages using
the table-fidelity prompt, a 5,000-character normalized OCR cap, and 1,536px images. It contains
4,505 valid outcomes, 79 auto-ties, three parse failures, and no detectable sentinel comparisons.
No judge calls were made during this replay.

## Result

All adaptive variants exhausted all ten sample batches. As on Britannica, savings came from
judging fewer pairs per batch rather than satisfying the stopping rule early.

| Strategy | Used | Saved | Kendall τ | Spearman ρ | Top-3 members/order | Median/max abs(ΔELO) | Final stat/practical/unresolved |
|---|---:|---:|---:|---:|---|---:|---:|
| Full / balanced | 4,505 | 0.0% | 1.000 | 1.000 | 3/3, yes | 0.0 / 0.0 | 2/0/11 |
| Targeted | 996 | 77.9% | 0.846 | 0.960 | 3/3, no | 46.2 / 115.7 | 1/0/12 |
| Targeted + 3×, min 10 | 803 | 82.2% | 0.780 | 0.925 | 3/3, no | 45.9 / 114.8 | 1/5/7 |
| Warm-up 10 | 1,777 | 60.6% | 0.868 | 0.965 | 2/3, no | 31.9 / 73.0 | 2/0/11 |
| Explore every 3 | 2,167 | 51.9% | 0.824 | 0.916 | 3/3, yes | 15.6 / 70.8 | 2/0/11 |

No strategy passed the original gate. Size-aware stopping saved another 193 outcomes relative to
plain targeted but further reduced full-order agreement. The size annotations remain practical
deployment preferences, not statistical resolutions.

### Rank movement

The full top three are:

1. `lightonai/LightOnOCR-2-1B`
2. `deepseek-ai/DeepSeek-OCR-2`
3. `baidu/Qianfan-OCR`

Plain targeted preserves membership but swaps ranks 2 and 3. The 10-model warm-up loses one
full-board top-3 member. Periodic exploration restores the exact top-three order at the cost of
more than doubling targeted's outcome count, while full-order Kendall τ remains only 0.824.

Per-model ELO changes are in [`elo-deltas.csv`](elo-deltas.csv).

## Page-clustered stability

The cross-board script resampled complete pages 200 times and reran each fixed allocation over five
seeds. The full 4,505-outcome board itself has:

- median page-bootstrap Kendall τ: **0.780**;
- top-3 membership stability: **19.0%**;
- exact top-3 order stability: **10.5%**; and
- 51 robust pair directions out of 91 (direction repeated in ≥95% of page bootstraps).

The leaders are therefore not stably ordered across page samples even when every stored comparison
is used. This is not a problem that a stopping threshold can solve.

At a 40% fixed budget (1,802 outcomes):

| Design | Clustered Spearman ρ | Top-3 membership agreement | Robust-pair agreement | Median/p95 max abs(ΔELO) |
|---|---:|---:|---:|---:|
| Pair-balanced | 0.960 | 47.4% | 98.4% | 34.3 / 55.8 |
| Mixed-random | 0.949 | 37.2% | 98.3% | 41.0 / 68.2 |

Both designs preserve the board's robust pairwise directions well, but neither can reproduce an
unstable top-three set. At 60%, robust-pair agreement rises to roughly 98.9%, while top-3 agreement
remains only 61.5%/56.1%.

## Conclusion

This independent large-grid board confirms the Britannica warning more strongly:

- targeted allocation is deterministic and cheap, but does not preserve the full board closely;
- ordinary adjacent marginal CIs do not describe page-sampling instability;
- practical size preferences should not control statistical sampling; and
- exact ranks among overlapping leaders are the wrong estimand.

**Recommendation:** do not proceed to a live targeted trial or change the balanced default. Stop
allocation tuning until the product can report uncertainty-aware rank tiers or robust pairwise
relations using page-clustered uncertainty. Any such production change needs its own design and
validation work; this experiment should not directly change behavior.

## Reproduce

```bash
uv run python experiments/adaptive-stopping/replay.py \
  --repo davanstrien/ocr-bench-moh-results \
  --revision 05d24695f131432c38984dd2cb55f5a18a3c2580 \
  --output-dir experiments/adaptive-stopping/moh

uv run python experiments/adaptive-stopping/multi_board.py
```
