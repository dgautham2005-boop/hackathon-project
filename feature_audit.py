"""feature_audit.py -- Person 1: data audit + improved causal features.

Standalone. Does NOT modify features.py / starter.py / solution.py / starter_model.json
(those stay as the working fallback). Same public surface as features.py so that
integration is `from feature_audit import predict`:

    matrix(obs)           -> (n_customers, N_FEATURES) float array, rows in obs['customers'] order
    fit(cases, processor) -> JSON-serialisable model dict (same keys as the starter: mean/scale/coefficients)
    predict(obs, model)   -> {'predictions': {...}, 'selected': [...]}

Extra helpers: clean_history(), audit_counts(), run_self_tests().

Rules implemented (BRIEF.md / EVENT_RULES.md / OBSERVATIONS.md):
  * purchase line  = CustomerID present AND InvoiceNo present AND InvoiceNo not starting 'C'/'c'
                     AND Quantity > 0 AND UnitPrice > 0
  * exact duplicate = all eight original fields equal (source_row ignored) -> counted once
  * only rows with history_start <= InvoiceDate < observed_until are used (the 7 days before the
    snapshot are unavailable, NOT zero demand)
  * returns/cancellations (C invoices, or negative quantity) are evidence features only
  * no labels, no future rows, no customer-ID values are ever used as features
Only numpy + stdlib are used (no pickle/joblib).
"""
from datetime import datetime
import math
import numpy as np

EIGHT = ('InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
         'UnitPrice', 'CustomerID', 'Country')
DEFAULT_COLUMNS = EIGHT + ('source_row',)
CAP = 2000.0
RIDGE = 10.0

# (name, transform) -- transform 'log' => log1p(max(0,x)) applied in _vector()
_FEATURES = [
    # --- the starter's eight informative features, redefined on cleaned data / observed_until clock
    ('recency', None),              # days from last purchase invoice to observed_until
    ('invoice_frequency', 'log'),   # distinct purchase invoices
    ('recent_value', 'log'),        # spend in the 30 observed days before observed_until
    ('older_value', 'log'),         # spend before that
    ('basket_value', 'log'),        # mean invoice value
    ('return_ratio', None),         # return value / purchase value (capped)
    ('cadence', None),              # mean gap between invoices (days), censored default
    ('tenure', None),               # days from first purchase invoice to observed_until
    # --- new
    ('has_history', None),
    ('purchase_lines', 'log'),
    ('active_days', 'log'),
    ('total_spend', 'log'),
    ('max_invoice_value', 'log'),
    ('median_invoice_value', 'log'),
    ('spend_60', 'log'),
    ('spend_90', 'log'),
    ('invoices_30', 'log'),
    ('invoices_90', 'log'),
    ('invoice_trend', None),        # (inv last90 - inv prior90)/(sum+1), in [-1,1]
    ('spend_trend', None),          # same on spend
    ('recent_share', None),         # spend_30 / total_spend
    ('rate_30d', None),             # invoices per 30 observed days
    ('return_value', 'log'),
    ('return_invoices', 'log'),
    ('return_share', None),         # return invoices / (return + purchase invoices)
    ('days_since_return', None),
    ('gap_median', None),
    ('gap_std', None),
    ('last_gap', None),
    ('overdue_ratio', None),        # recency / cadence (>1 = overdue), clipped
    ('has_gap', None),              # >=2 invoices so gap stats are real
    ('avg_qty_line', 'log'),
    ('avg_line_value', 'log'),
]
FEATURE_NAMES = tuple(n for n, _ in _FEATURES)
N_FEATURES = len(FEATURE_NAMES)
_LOG_IDX = [i for i, (_, t) in enumerate(_FEATURES) if t == 'log']
ORIGINAL_EQUIVALENT = FEATURE_NAMES[:8]   # starter's 9th feature (observed_gap) is constant 7 -> dropped


def _parse(ts):
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _col_index(obs):
    cols = obs.get('transactions', {}).get('columns') or list(DEFAULT_COLUMNS)
    return {c: i for i, c in enumerate(cols)}


def clean_history(obs, counts=None):
    """Return {customer: {'purchases': [...], 'returns': [...]}} for eligible customers.

    purchases: (date, invoice, quantity, price) ; returns: (date, abs_value, is_C_invoice, invoice_or_None)
    `counts` (dict) is filled with audit tallies if supplied."""
    c = counts if counts is not None else {}
    idx = _col_index(obs)
    ids = set(obs['customers'])
    start, end = _parse(obs.get('history_start')) or datetime.min, _parse(obs['observed_until'])
    out = {cid: {'purchases': [], 'returns': []} for cid in ids}
    seen = set()
    for r in obs['transactions']['rows']:
        c['rows'] = c.get('rows', 0) + 1
        key = tuple(r[idx[k]] for k in EIGHT)          # source_row excluded on purpose
        if key in seen:
            c['exact_duplicates'] = c.get('exact_duplicates', 0) + 1
            continue
        seen.add(key)
        inv, _, _, qty, date, price, cid, _ = key
        if cid is None:
            c['missing_customer'] = c.get('missing_customer', 0) + 1
            continue
        if cid not in ids:
            c['non_roster_customer'] = c.get('non_roster_customer', 0) + 1
            continue
        d = _parse(date)
        if d is None or qty is None or price is None or isinstance(qty, bool) or isinstance(price, bool):
            c['unusable_row'] = c.get('unusable_row', 0) + 1
            continue
        if not (start <= d < end):
            c['outside_window'] = c.get('outside_window', 0) + 1
            continue
        inv_s = None if inv is None else str(inv).strip()
        is_c = bool(inv_s) and inv_s[0] in 'Cc'
        if inv_s and not is_c and qty > 0 and price > 0:
            out[cid]['purchases'].append((d, inv_s, float(qty), float(price)))
            c['purchase_lines'] = c.get('purchase_lines', 0) + 1
        elif is_c or qty < 0:
            out[cid]['returns'].append((d, abs(qty * price) if price > 0 else 0.0, is_c, inv_s))
            c['return_lines'] = c.get('return_lines', 0) + 1
        else:
            c['ignored_nonpurchase'] = c.get('ignored_nonpurchase', 0) + 1   # null/zero price, null invoice, qty==0
    return out


def audit_counts(obs):
    c = {}
    clean_history(obs, c)
    return c


def _customer_vector(rec, ref, window_days):
    W = max(window_days, 30.0)
    inv = {}
    qty_sum = val_sum = 0.0
    dates = set()
    for d, i, q, p in rec['purchases']:
        age = (ref - d).total_seconds() / 86400.0
        e = inv.setdefault(i, [0.0, 1e9])
        e[0] += q * p
        e[1] = min(e[1], age)
        qty_sum += q
        val_sum += q * p
        dates.add(d.date())
    n = len(inv)
    values = sorted(v for v, _ in inv.values())
    ages = sorted(a for _, a in inv.values())
    total = sum(values)
    s = lambda lo, hi: sum(v for v, a in inv.values() if lo <= a < hi)
    k = lambda lo, hi: sum(1 for _, a in inv.values() if lo <= a < hi)
    gaps = [b - a for a, b in zip(ages, ages[1:])]
    cad = sum(gaps) / len(gaps) if gaps else W / 2.0
    rec_days = ages[0] if ages else W
    ret_val = sum(r[1] for r in rec['returns'])
    ret_inv = len({r[3] for r in rec['returns'] if r[2] and r[3]})   # distinct cancellation (C) invoices
    ret_ages = [(ref - r[0]).total_seconds() / 86400.0 for r in rec['returns']]
    inv30, inv90, inv_prev = k(0, 30), k(0, 90), k(90, 180)
    sp30, sp60, sp90, sp_prev = s(0, 30), s(0, 60), s(0, 90), s(90, 180)
    med = values[len(values) // 2] if values else 0.0
    gstd = float(np.std(gaps)) if len(gaps) >= 2 else 0.0
    return [
        rec_days, n, sp30, total - sp30, total / n if n else 0.0,
        min(ret_val / max(1.0, total), 20.0), cad, ages[-1] if ages else W,
        1.0 if n else 0.0, len(rec['purchases']), len(dates), total,
        values[-1] if values else 0.0, med, sp60, sp90, inv30, inv90,
        (inv90 - inv_prev) / (inv90 + inv_prev + 1.0), (sp90 - sp_prev) / (sp90 + sp_prev + 1.0),
        sp30 / total if total > 0 else 0.0, n / W * 30.0,
        ret_val, ret_inv, ret_inv / max(1.0, ret_inv + n) if ret_inv else 0.0,
        min(ret_ages) if ret_ages else W,
        float(np.median(gaps)) if gaps else W / 2.0, gstd, gaps[0] if gaps else W / 2.0,
        min(rec_days / max(cad, 1.0), 10.0) if n >= 2 else 0.0, 1.0 if n >= 2 else 0.0,
        qty_sum / len(rec['purchases']) if rec['purchases'] else 0.0,
        val_sum / len(rec['purchases']) if rec['purchases'] else 0.0,
    ]


def matrix(obs):
    ids = obs['customers']
    if 'prepared_features' in obs:
        f = obs['prepared_features']
        if not isinstance(f, dict) or set(f) != set(ids):
            raise ValueError('Feature customer identity mismatch')
        m = np.asarray([f[c] for c in ids], dtype=float)
        if m.shape != (len(ids), N_FEATURES) or not np.isfinite(m).all():
            raise ValueError('Expected %d finite causal features per customer' % N_FEATURES)
        return m
    ref = _parse(obs['observed_until'])
    start = _parse(obs.get('history_start'))
    window = (ref - start).total_seconds() / 86400.0 if start else 180.0
    hist = clean_history(obs)
    m = np.asarray([_customer_vector(hist[c], ref, window) for c in ids], dtype=float)
    if m.shape != (len(ids), N_FEATURES) or not np.isfinite(m).all():
        raise ValueError('Invalid causal features')
    return m


def _vector(obs):
    a = matrix(obs).copy()
    a[:, _LOG_IDX] = np.log1p(np.maximum(0.0, a[:, _LOG_IDX]))
    return np.clip(a, -1e6, 1e6)


def fit(cases, processor, ridge=RIDGE):
    xs, ys = [], []
    for case in cases:
        obs = processor('aiml', case['observation'])
        if obs['customers'] != case['observation']['customers']:
            raise ValueError('Processor changed eligible customers')
        xs.append(_vector(obs))
        ys.extend([[case['targets'][c][0], case['targets'][c][1] / CAP] for c in obs['customers']])
    x, y = np.concatenate(xs), np.asarray(ys, dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    pen = np.eye(x.shape[1]) * ridge          # dimension follows the feature count (starter hard-coded 10)
    pen[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': coef.tolist(),
            'feature_names': list(FEATURE_NAMES)}


def predict(obs, model):
    if len(model['mean']) != N_FEATURES:
        raise ValueError('model has %d features, feature_audit expects %d' % (len(model['mean']), N_FEATURES))
    x = (_vector(obs) - np.asarray(model['mean'])) / np.asarray(model['scale'])
    est = np.clip(np.column_stack([np.ones(len(x)), x]) @ np.asarray(model['coefficients']), 0.0, 1.0)
    preds = {c: {'probability': float(p[0]), 'value': float(p[1] * CAP)} for c, p in zip(obs['customers'], est)}
    selected = sorted(preds, key=lambda c: (-preds[c]['value'], c))[:obs['budget']]
    return {'predictions': preds, 'selected': selected}


# ----------------------------------------------------------------------------- self-tests
def run_self_tests(obs):
    """Invariance / leakage tests on one raw observation. Raises AssertionError on failure."""
    import copy, random
    base = matrix(obs)
    rows = obs['transactions']['rows']
    idx = _col_index(obs)
    # 1. exact duplicates are ignored; adding copies with a different source_row must not change anything
    o = copy.deepcopy(obs)
    extra = []
    for r in rows[:50]:
        r2 = list(r); r2[idx['source_row']] = 'dup' + str(r[idx['source_row']]); extra.append(r2)
    o['transactions']['rows'] = rows + extra
    assert np.array_equal(matrix(o), base), 'duplicate rows changed features'
    # 2. future rows (>= observed_until, incl. the target window) must not change anything
    o = copy.deepcopy(obs)
    fut = []
    for r in rows[:50]:
        r2 = list(r); r2[idx['InvoiceDate']] = obs['observed_until'][:10] + 'T12:00:00'; r2[idx['InvoiceNo']] = '9' + str(r[idx['InvoiceNo']]); fut.append(r2)
    o['transactions']['rows'] = rows + fut
    assert np.array_equal(matrix(o), base), 'future rows leaked into features'
    # 3. unknown-customer rows and non-roster customers are ignored
    o = copy.deepcopy(obs)
    junk = []
    for r in rows[:50]:
        r2 = list(r); r2[idx['CustomerID']] = None; junk.append(r2)
        r3 = list(r); r3[idx['CustomerID']] = 'NOT-A-CUSTOMER'; junk.append(r3)
    o['transactions']['rows'] = rows + junk
    assert np.array_equal(matrix(o), base), 'unknown customer rows leaked'
    # 4. row order invariance
    o = copy.deepcopy(obs); sh = list(rows); random.Random(0).shuffle(sh); o['transactions']['rows'] = sh
    assert np.allclose(matrix(o), base), 'row order changed features'
    # 5. customer-ID relabelling invariance (no memorised IDs): features follow the history, not the label
    ids = obs['customers']; perm = {c: 'X%05d' % i for i, c in enumerate(reversed(ids))}
    o = copy.deepcopy(obs); o['customers'] = [perm[c] for c in ids]
    o['transactions']['rows'] = [[(perm[v] if k == 'CustomerID' and v in perm else v) for k, v in zip(DEFAULT_COLUMNS, r)] for r in rows]
    assert np.allclose(matrix(o), base), 'features depend on customer id values'
    return True


if __name__ == '__main__':
    import argparse, gzip, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='training.json.gz')
    ap.add_argument('--audit', action='store_true', help='print data-quality counts per case (totals)')
    ap.add_argument('--selftest', action='store_true', help='run leakage/invariance tests on the first case')
    ap.add_argument('--train', metavar='OUT_JSON', help='fit ridge on --data and write JSON model (no pickle)')
    ap.add_argument('--ridge', type=float, default=RIDGE)
    a = ap.parse_args()
    cases = json.loads(gzip.decompress(Path(a.data).read_bytes()))
    if a.audit:
        tot = {}
        for c in cases:
            for k, v in audit_counts(c['observation']).items():
                tot[k] = tot.get(k, 0) + v
        print(json.dumps(tot, indent=1))
    if a.selftest:
        for c in cases[:5]:
            run_self_tests(c['observation'])
        print('self-tests passed on 5 cases')
    if a.train:
        Path(a.train).write_text(json.dumps(fit(cases, lambda d, o: o, a.ridge)) + '\n')
        print('wrote', a.train)
