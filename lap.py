"""
Compatibility shim so that `from lap import lapjv` works for Ultralytics trackers.
Priority:
1) Use lapx.lapjv if available
2) Use lap.lapjv if available
3) Fallback to scipy.optimize.linear_sum_assignment to emulate lap.lapjv
"""

import numpy as _np

_lapjv_impl = None

# 1) Try lapx
try:
    from lapx import lapjv as _lapx_lapjv  # type: ignore
    _lapjv_impl = _lapx_lapjv
except Exception:
    _lapjv_impl = None

# 2) Try real lap
if _lapjv_impl is None:
    try:
        import lap as _lap_mod  # type: ignore
        if hasattr(_lap_mod, "lapjv"):
            _lapjv_impl = getattr(_lap_mod, "lapjv")
    except Exception:
        _lapjv_impl = None

# 3) Fallback to scipy implementation
if _lapjv_impl is None:
    try:
        from scipy.optimize import linear_sum_assignment as _lsa

        def _scipy_lapjv(cost_matrix, extend_cost=True, cost_limit=1e9, return_cost=False):
            C = _np.asarray(cost_matrix, dtype=_np.float64)
            # emulate cost_limit behavior
            C_mod = C.copy()
            if cost_limit is not None:
                C_mod[C_mod > cost_limit] = cost_limit

            r, c = _lsa(C_mod)
            n_rows, n_cols = C_mod.shape
            x = _np.full(n_rows, -1, dtype=_np.int64)
            y = _np.full(n_cols, -1, dtype=_np.int64)
            for rr, cc in zip(r, c):
                # only accept assignments under cost_limit
                if cost_limit is None or C_mod[rr, cc] <= cost_limit:
                    x[rr] = int(cc)
                    y[cc] = int(rr)
            opt = float(C_mod[r, c].sum()) if len(r) else 0.0
            if return_cost:
                return opt, x, y, opt
            return opt, x, y

        _lapjv_impl = _scipy_lapjv
    except Exception:
        pass


def lapjv(cost_matrix, extend_cost=True, cost_limit=1e9, return_cost=False):
    if _lapjv_impl is None:
        raise ImportError("No available LAP solver (lap/lapx/scipy) for lap.lapjv shim")
    return _lapjv_impl(cost_matrix, extend_cost=extend_cost, cost_limit=cost_limit, return_cost=return_cost)

