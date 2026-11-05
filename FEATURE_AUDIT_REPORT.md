# Feature audit report (Person 1) — repeat purchase / customer value

Scope: audit of `features.py` (cleaning + 9 features), plus an improved, leakage-safe feature module
`feature_audit.py`. **No original file was modified** (sha256 of features.py, starter.py, solution.py,
starter_model.json match the uploaded zip). All numbers below were produced locally with the public
training (75 cases) and practice (51 cases) files; practice is public, so treat it as development data.

## 1. Problems found in the original features

| # | Problem | Evidence | Impact |
|---|---------|----------|--------|
| 1 | **Exact duplicate rows are never removed** (starter groups raw rows). | 2,530 extra rows in training (1.6%), 2,331 in practice (2.9%). Appending copies of 50 rows with new `source_row` changes starter features; it does not change `feature_audit` features. | Inflates frequency/value/basket; target scoring counts duplicates once, so features and target disagree. |
| 2 | **Feature 9 `observed_gap` is the constant 7.0.** | Single unique value across all 75 training cases. | Dead column (scale forced to 1, no information); wastes a coefficient. |
| 3 | **"Recent" window is mislabelled.** `age <= 30` is measured from the snapshot, but the last 7 days before it are unobserved (history ends at `observed_until`). | OBSERVATIONS.md; gap is always 7 days. | "Recent" is really 23 observed days; recent/older split is shifted. `feature_audit` measures every window from `observed_until`. |
| 4 | **`fit()` hard-codes `np.eye(10)`.** | features.py `penalty = np.eye(10)*RIDGE`. | Any change in feature count crashes training. Patched to `x.shape[1]`. |
| 5 | **Arbitrary defaults** (recency 180, cadence 90, tenure 180) with no indicator. | In public data every rostered customer has ≥1 purchase, so this is dormant, but hidden cohorts may differ. | "No data" is indistinguishable from real values. Added `has_history`, `has_gap`. |
| 6 | Dates compared as **strings** (`history_start <= r[4] < observed_until`). | Works today (all timestamps are 19-char ISO). | Fragile if a format ever varies; `feature_audit` parses dates. |
| 7 | `return_ratio` is unbounded. | returns / max(1, purchases). | Outliers; now capped at 20. |
| 8 | Shortlist ranks by predicted value only. | `predict()` sorts by `-value`. | Tested p×value ranking: no consistent gain (see §3). Left as is; modelling choice for Person 2/3. |

## 2. Rules verified (checklist 1–7)

* **Cleaning / qualifying purchase** (CustomerID and InvoiceNo present, no `C`/`c` prefix, Quantity>0, UnitPrice>0): the starter implements this correctly (it uses `.upper()`, so lowercase `c` is handled; null invoices are never purchases). `feature_audit` implements it identically, with explicit tallies via `audit_counts()`.
* **Exact duplicates:** keyed on the eight original fields, `source_row` excluded — confirmed `source_row` is unique per row (so including it would hide every duplicate). Keep-first.
* **Cancellations/returns:** `C` invoices and negative quantities are return *evidence* only; they never count as purchases. 4,016 return lines (training) / 1,462 (practice) after dedup.
* **Missing CustomerID / InvoiceNo:** ignored for the roster join (never inferred). **The public data contains none** (0 null CustomerID, 0 null InvoiceNo, 0 lowercase `c`, 0 non-`C` negative quantities, 0 `C` rows with positive quantity, 0 rows outside the window); the code handles them anyway, tested with injected rows. Hidden cohorts may differ — this is untested on real hidden data.
* **Nonpositive price:** only 15 rows (10 training / 5 practice); ignored as purchases.
* **Time safety:** only `history_start <= date < observed_until` is used. Test: injecting rows dated at/after `observed_until` (including the target window) leaves features byte-identical. No labels are read in `matrix()`/`predict()`; labels appear only in `fit()` as targets.
* **No ID memorisation:** relabelling all customer IDs leaves features unchanged (self-test 5); IDs are used only as join keys.
* Self-tests: `python feature_audit.py --selftest` (duplicates, future rows, unknown customers, row order, ID relabelling).

## 3. Results (same ridge learner unless stated; local formula = official checker)

| Feature set | Practice score | Brier | Value MSE | Capture | Grouped-CV score (train, by snapshot) |
|---|---|---|---|---|---|
| A. Starter features (reproduces baseline_results 80.21) | 80.21 | 0.1608 | 72,833 | 0.573 | 80.18 |
| B. Starter features + dedup (the patch) | **80.61** | 0.1606 | 72,613 | 0.586 | 80.14 |
| C. `feature_audit` first 8 (cleaned, observed_until clock) | 80.53 | 0.1604 | 72,405 | 0.583 | 80.11 |
| D. All 33 features, ridge λ=300 | 79.40 | 0.1585 | 72,520 | 0.543 | 80.45 |
| D. All 33 features, ridge λ=1000 | 80.09 | 0.1588 | 72,622 | 0.566 | 80.17 |
| Diagnostic, HistGradientBoosting, starter features | 79.83 | 0.1604 | 66,304 | 0.558 | — |
| Diagnostic, HistGradientBoosting, 33 features | 80.58 | 0.1603 | 68,339 | 0.583 | — |

**Honest reading:** the gains are small and not clearly outside noise. Practice has 51 cases and ~8 shortlisted
customers per case, so capture swings by several points on small changes (fixing one feature definition moved the
HGB number by ~0.5). Dedup is a correctness fix worth making but its score effect is +0.4 on practice and ≈0 in CV.
The 33 features lower Brier modestly under ridge (0.1608→0.1585) but a *linear* model on the heavy-tailed value target
hurts selection, so **do not swap in the extended features with the starter ridge and expect a higher score.** I have
not run significance tests or bootstrap intervals. Diagnostic HGB used local scikit-learn 1.9.1 (the grader pins 1.7.2);
it is not part of any deliverable.

Univariate AUC for repeat purchase (train / practice): active_days 0.70/0.67, total_spend 0.70/0.69,
invoice_frequency 0.70/0.68, spend_90 0.70/0.70, recency 0.33/0.32 (i.e. 0.67 inverted). **`tenure` is unstable
(0.58 train → 0.50 practice)** — consider dropping it. `has_history` is constant in public data.
Activity volume and recency carry most of the signal; trend and return features add little on their own.

## 4. Recommended improvements

1. Apply the duplicate-removal patch (safe, tiny, dimension-preserving).
2. Use `feature_audit.matrix()` as the feature source for a **nonlinear / two-part learner** (probability and value fitted separately; value is heavy-tailed, 1.45% of customers hit the £2,000 cap), selected by grouped temporal CV by snapshot. Linear ridge alone is not enough to exploit them.
3. Drop or regularise `tenure` and the constant/degenerate columns (`has_history`) if the model is sensitive.
4. Keep ranking by predicted value unless Person 2/3 show p×value is better in CV (my test: no consistent gain).
5. Use grouped folds by snapshot (`dataset_info.json` group_index) — customers/snapshots overlap otherwise.

## 5. Files

Changed: **none of the originals.** New:
* `feature_audit.py` — cleaning, 33 causal features, drop-in `matrix/fit/predict`, self-tests, CLI (`--audit`, `--selftest`, `--train OUT.json`). numpy + stdlib only; no pickle/joblib.
* `features_patch.diff` — minimal patch for `features.py` (unified diff vs the starter).
* `solution_audit_example.py` — example wiring (needs a `model_audit.json` you train).
* `audit_experiments.py` — reproduces the table in §3 (`python audit_experiments.py`).
* `FEATURE_AUDIT_REPORT.md` — this file.

## 6. Integration guide for Person 4

**Public signatures are unchanged** in both options: `matrix(obs)`, `fit(cases, processor)`, `predict(obs, model)`.
`feature_audit.fit` adds an optional keyword `ridge=` (backward compatible) and the model dict gains an extra `feature_names` key
(ignored by the starter's `predict`).

**Option A — minimal (patch):** `patch features.py < features_patch.diff`, then `python train.py` and
`python local_check.py ...`. Feature count stays 9, so `starter_model.json` still *loads*, but it was fitted on
non-deduplicated features: **retrain it**. Keep a copy of the original features.py as fallback.

**Option B — extended features:** put `feature_audit.py` at the ZIP root and import from it
(`from feature_audit import predict` in solution.py; see `solution_audit_example.py`). Consequences:
* Feature dimension 9 → **33**. Model JSON must have 33 `mean`/`scale` entries and `coefficients` of shape **34×2** (intercept + 33, columns = probability, value/2000). The old `starter_model.json` **will not work**: `predict()` raises a ValueError on a dimension mismatch (deliberate guard).
* `train.py` must import `fit` from `feature_audit` (or call `python feature_audit.py --data training.json.gz --train model.json --ridge 300`).
* `obs['prepared_features']`, if ever supplied, must have 33 columns.
* If you use another learner: call `feature_audit._vector(obs)` for log-transformed features (columns listed in `FEATURE_NAMES`) or `matrix(obs)` for raw ones; rows follow `obs['customers']`. Tree/GBM parameters must be exported to JSON (no pickle/joblib) and the inference code reimplemented in numpy.
* Feature extraction costs ~5 ms per case; well inside the 120 s limit.
* The ZIP may include `.py/.json/.md/.txt` only — do not ship the data files.
