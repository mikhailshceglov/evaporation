#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_awat.py

AWAT (Hess et al., 2014) pipeline:
1) Read CSV (sep=';', decimal=',', utf-8-sig)
2) Parse datetime column, sort
3) Split by gaps (gap_minutes)
4) For each point i:
   - take window w_max (odd, clipped at edges)
   - select polynomial degree k by AICc (k=0..poly_max_k, and k<=m-2)
   - compute s_res,i and s_dat,i
   - B_i = s_res,i / s_dat,i = sqrt(1 - R_i^2)
   - adaptive window w_i = max(w_min, B_i*w_max) rounded to odd, clipped to [w_min, w_max]
5) Adaptive moving average with w_i => tilde_y
6) Thresholding with memory:
      z_0 = tilde_y_0
      z_i = z_{i-1} if |tilde_y_i - z_{i-1}| < delta_i else tilde_y_i
   where delta_i = clip(s_res,i * t_{0.975, df}, delta_min, delta_max),
   df = max(1, m - (k+1)) (residual dof in the fit window)
7) noise_i = |y_i - z_i|
8) Noise intervals: consecutive points with noise_i > noise_eps (broken by time gaps too),
   length >= min_noise_len. Save one PNG per interval.

Only: pandas, numpy, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# -----------------------------
# Helpers
# -----------------------------
def _oddify(n: int) -> int:
    n = int(n)
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _pick_tick_indices(n: int, max_ticks: int = 8) -> np.ndarray:
    """
    Return indices for x-axis ticks. Ensures ticks correspond to actual data points.
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n <= max_ticks:
        return np.arange(n, dtype=int)
    # include first & last
    idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    idx = np.unique(idx)
    return idx.astype(int)


def _fmt_dt(dt: pd.Timestamp) -> str:
    # stable filename-safe-ish
    return dt.strftime("%Y%m%d_%H%M%S")


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    return s.strip("_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _auto_detect_y_column(df: pd.DataFrame, dt_col: str) -> str:
    """
    Heuristic y auto-selection for various schemas.
    Prefers:
      - "level_cleaned" / "level" / "уровень" / "мм"
    Penalizes:
      - temperature, voltage, battery
    """
    best_col = None
    best_score = -1e18

    for c in df.columns:
        if c == dt_col:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        cnt = int(s.notna().sum())
        if cnt == 0:
            continue

        name = str(c).strip().lower()

        score = float(cnt)
        # strong preferences
        if "level_cleaned" in name:
            score += 2_000_000
        if "level" in name:
            score += 800_000
        if "уров" in name or "уровень" in name:
            score += 800_000
        if "мм" in name:
            score += 200_000

        # penalize non-level columns
        if "температ" in name or "temp" in name:
            score -= 300_000
        if "напряж" in name or "акб" in name or "volt" in name or "battery" in name:
            score -= 300_000

        # prefer smoother signal: smaller median abs diff (only if enough points)
        if cnt >= 50:
            vals = s.dropna().to_numpy(dtype=float)
            d = np.abs(np.diff(vals))
            if d.size > 0:
                score -= float(np.median(d)) * 1_000  # weak tie-breaker

        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError("Cannot auto-detect y column. Use --y-col explicitly.")
    return str(best_col)


@dataclass
class FitResult:
    k: int
    s_res: float
    s_dat: float
    B: float
    delta: float


class PolyFitCache:
    """
    Cache for (m,k) -> (x, V, pinv(V)) where
      x: length m
      V: Vandermonde with columns x^0..x^k (increasing=True), shape (m, k+1)
      pinv: shape (k+1, m)
    Works for both odd and even m (edges may produce even windows).
    """
    def __init__(self):
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, m: int, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = int(m)
        k = int(k)
        key = (m, k)
        if key in self._cache:
            return self._cache[key]

        # FIX: x must be length exactly m even when m is even
        # center around 0; for even m we get half-step centered grid
        x = np.arange(m, dtype=float) - (m - 1) / 2.0  # length m

        V = np.vander(x, N=k + 1, increasing=True)      # (m, k+1)
        pinv = np.linalg.pinv(V)                        # (k+1, m)

        self._cache[key] = (x, V, pinv)
        return x, V, pinv

def _aicc(rss: float, n: int, p: int) -> float:
    """
    AICc for least squares:
      AIC = n * ln(RSS/n) + 2p
      AICc = AIC + 2p(p+1)/(n-p-1)
    """
    n = int(n)
    p = int(p)
    if n <= 0:
        return float("inf")
    if rss <= 0:
        # perfect fit
        return -1e30
    if n - p - 1 <= 0:
        return float("inf")
    aic = n * math.log(rss / n) + 2 * p
    aicc = aic + (2 * p * (p + 1)) / (n - p - 1)
    return float(aicc)

def _fit_window(
    y_win: np.ndarray,
    m: int,
    w_max: int,
    poly_max_k: int,
    delta_min: float,
    delta_max: float,
    cache: PolyFitCache,
) -> FitResult:
    y_win = np.asarray(y_win, dtype=float)
    m = int(len(y_win))  # FIX: trust actual window length
    if m < 3:
        # too short: fallback
        s_dat = float(np.std(y_win, ddof=1)) if m >= 2 else 0.0
        s_res = 0.0
        B = 0.0
        delta = _clip(0.0, delta_min, delta_max)
        return FitResult(k=0, s_res=s_res, s_dat=s_dat, B=B, delta=delta)

    s_dat = float(np.std(y_win, ddof=1))
    if not np.isfinite(s_dat) or s_dat <= 0:
        s_dat = 0.0

    # bounds for k: need m - (k+1) >= 1 and AICc denom n-p-1 positive => m-(k+1)-1 >0 => m-k-2>0 => k<=m-3
    k_max = min(int(poly_max_k), m - 3)
    if k_max < 0:
        k_max = 0

    best_k = 0
    best_aicc = float("inf")
    best_rss = float("inf")
    best_p = 1
    best_yhat = None

    for k in range(0, k_max + 1):
        _, V, pinv = cache.get(m, k)
        coeff = pinv @ y_win
        y_hat = V @ coeff
        resid = y_win - y_hat
        rss = float(np.sum(resid * resid))
        p = k + 1
        aicc = _aicc(rss=rss, n=m, p=p)
        if aicc < best_aicc:
            best_aicc = aicc
            best_k = k
            best_rss = rss
            best_p = p
            best_yhat = y_hat

    if best_yhat is None:
        best_yhat = np.full_like(y_win, float(np.mean(y_win)))
        best_rss = float(np.sum((y_win - best_yhat) ** 2))
        best_k = 0
        best_p = 1

    # residual std with dof correction
    dof = max(1, m - best_p)
    s_res = math.sqrt(best_rss / dof) if best_rss >= 0 else 0.0

    # R^2
    y_mean = float(np.mean(y_win))
    sst = float(np.sum((y_win - y_mean) ** 2))
    if sst <= 0:
        r2 = 1.0
    else:
        r2 = 1.0 - (best_rss / sst)
        r2 = float(np.clip(r2, -1.0, 1.0))

    # B = sqrt(1 - R^2) (>=0)
    B = math.sqrt(max(0.0, 1.0 - r2))

    # delta_i = clip(s_res * t_{0.975, dof}, delta_min, delta_max)
    tcrit = float(student_t.ppf(0.975, df=dof)) if dof > 0 else 1.96
    delta = _clip(s_res * tcrit, delta_min, delta_max)

    return FitResult(k=best_k, s_res=float(s_res), s_dat=float(s_dat), B=float(B), delta=float(delta))


def _compute_segment_awat(
    dt: np.ndarray,
    y: np.ndarray,
    w_max: int,
    w_min: int,
    delta_min: float,
    delta_max: float,
    poly_max_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute z, s_res, B, w_i, delta for one continuous segment (no large time gaps inside).
    """
    n = len(y)
    y = np.asarray(y, dtype=float)

    w_max = _oddify(w_max)
    w_min = _oddify(w_min)
    w_min = min(w_min, w_max)

    cache = PolyFitCache()

    s_res = np.full(n, np.nan, dtype=float)
    B = np.full(n, np.nan, dtype=float)
    w_i = np.full(n, w_min, dtype=int)
    delta_i = np.full(n, np.nan, dtype=float)

    # 1) local fits (AICc -> k, s_res, B, delta) on window w_max
    for i in range(n):
        h = w_max // 2
        l = max(0, i - h)
        r = min(n - 1, i + h)
        # ensure odd length by trimming if needed
        m = r - l + 1
        if m % 2 == 0:
            if r < n - 1:
                r += 1
            elif l > 0:
                l -= 1
            m = r - l + 1
        y_win = y[l:r + 1]
        res = _fit_window(
            y_win=y_win,
            m=m,
            w_max=w_max,
            poly_max_k=poly_max_k,
            delta_min=delta_min,
            delta_max=delta_max,
            cache=cache,
        )
        s_res[i] = res.s_res
        B[i] = res.B
        delta_i[i] = res.delta

        # w_i = max(w_min, B_i * w_max) rounded to odd, clipped
        if np.isfinite(res.B):
            w_val = res.B * w_max
        else:
            w_val = float(w_min)
        w_use = int(round(w_val))
        w_use = _oddify(w_use)
        w_use = max(w_min, min(w_use, w_max))
        w_i[i] = w_use

    # 2) adaptive moving average using w_i => tilde_y
    prefix = np.cumsum(np.r_[0.0, y])  # len n+1
    tilde = np.full(n, np.nan, dtype=float)
    for i in range(n):
        h = int(w_i[i]) // 2
        l = max(0, i - h)
        r = min(n - 1, i + h)
        s = prefix[r + 1] - prefix[l]
        tilde[i] = s / (r - l + 1)

    # 3) thresholding with memory => z
    z = np.full(n, np.nan, dtype=float)
    z[0] = tilde[0]
    for i in range(1, n):
        prev = z[i - 1]
        if not np.isfinite(prev):
            prev = tilde[i - 1]
        if abs(tilde[i] - prev) < float(delta_i[i]):
            z[i] = prev
        else:
            z[i] = tilde[i]

    return z, s_res, B, w_i.astype(int), delta_i


def _find_intervals(
    dt: pd.Series,
    mask: np.ndarray,
    gap_minutes: float,
    min_len: int,
) -> List[Tuple[int, int]]:
    """
    Find consecutive True-intervals in mask, breaking at time gaps > gap_minutes.
    """
    n = len(mask)
    if n == 0:
        return []

    dt64 = dt.to_numpy(dtype="datetime64[ns]")
    gaps = np.zeros(n, dtype=bool)
    if n > 1:
        dmins = (dt64[1:] - dt64[:-1]) / np.timedelta64(1, "m")
        gaps[1:] = dmins > float(gap_minutes)

    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i in range(n):
        if gaps[i]:
            if start is not None:
                end = i - 1
                if end - start + 1 >= int(min_len):
                    intervals.append((start, end))
                start = None

        if bool(mask[i]):
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                if end - start + 1 >= int(min_len):
                    intervals.append((start, end))
                start = None

    if start is not None:
        end = n - 1
        if end - start + 1 >= int(min_len):
            intervals.append((start, end))

    return intervals


def plot_noise_interval(
    out_path: Path,
    dt: pd.Series,
    y: np.ndarray,
    z: np.ndarray,
    interval_id: int,
    l: int,
    r: int,
    max_xticks: int = 8,
) -> None:
    """
    Save one PNG per interval. X ticks are only at timestamps from data.
    """
    dt_seg = dt.iloc[l:r + 1].reset_index(drop=True)
    y_seg = y[l:r + 1]
    z_seg = z[l:r + 1]

    n = len(y_seg)
    if n <= 1:
        return

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)

    ax.plot(dt_seg.to_numpy(), y_seg, linewidth=1.0, label="y (raw)")
    ax.plot(dt_seg.to_numpy(), z_seg, linewidth=1.2, label="z (AWAT)")

    ax.set_title(
        f"Noise interval #{interval_id}: {dt_seg.iloc[0]} — {dt_seg.iloc[-1]}  (len={n})"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("level (mm)")

    # grid (клетчатый)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.6)

    # X ticks: only from existing datetimes
    tick_idx = _pick_tick_indices(n, max_ticks=max_xticks)
    tick_pos = dt_seg.iloc[tick_idx].to_numpy()
    tick_lbl = [pd.Timestamp(x).strftime("%Y-%m-%d\n%H:%M:%S") for x in dt_seg.iloc[tick_idx]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=0, ha="center")

    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # ====== CHANGED DEFAULTS: everything into script_dir/out/ ======
    default_out_dir = script_dir / "out"
    default_in = project_root / "data" / "data.csv"
    default_out = default_out_dir / "out.csv"
    default_plots = default_out_dir / "noise_plots"
    # =============================================================

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=str(default_in),
                   help="Input CSV path (default: data/data.csv)")
    p.add_argument("--out", dest="out_path", default=str(default_out),
                   help="Output CSV path (default: rain/out/out.csv)")
    p.add_argument("--plots-dir", dest="plots_dir", default=str(default_plots),
                   help="Directory for noise interval PNGs (default: rain/out/noise_plots/)")

    p.add_argument("--dt-col", dest="dt_col", default="Дата/время",
                   help="Datetime column name (default: 'Дата/время'). For cleaned_data.csv use 'datetime'.")
    p.add_argument("--y-col", dest="y_col", default=None,
                   help="Target y column name. If not set, auto-detects.")

    p.add_argument("--w-max", dest="w_max", type=int, default=31)
    p.add_argument("--w-min", dest="w_min", type=int, default=5)
    p.add_argument("--delta-min", dest="delta_min", type=float, default=0.017)
    p.add_argument("--delta-max", dest="delta_max", type=float, default=0.08)
    p.add_argument("--poly-max-k", dest="poly_max_k", type=int, default=5)

    p.add_argument("--noise-eps", dest="noise_eps", type=float, required=True,
                   help="Noise threshold in mm for interval detection: noise_i > noise_eps")
    p.add_argument("--min-noise-len", dest="min_noise_len", type=int, default=15,
                   help="Minimum interval length in points (default: 15)")
    p.add_argument("--gap-minutes", dest="gap_minutes", type=float, default=10.0,
                   help="Split series if time gap > this many minutes (default: 10)")

    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    plots_dir = Path(args.plots_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    _ensure_dir(out_path.parent)
    _ensure_dir(plots_dir)

    # read
    df = pd.read_csv(in_path, sep=";", decimal=",", encoding="utf-8-sig")
    dt_col = str(args.dt_col)
    if dt_col not in df.columns:
        raise ValueError(f"Column '{dt_col}' not found. Columns: {list(df.columns)}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    df = df.rename(columns={dt_col: "datetime"})

    if args.y_col is not None:
        y_col = str(args.y_col)
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found. Columns: {list(df.columns)}")
    else:
        y_col = _auto_detect_y_column(df, dt_col="datetime")

    y = pd.to_numeric(df[y_col], errors="coerce")
    df = df.assign(y=y).dropna(subset=["y"]).reset_index(drop=True)

    # gap split
    dt = df["datetime"]
    dt64 = dt.to_numpy(dtype="datetime64[ns]")
    n = len(df)

    gap_break = np.zeros(n, dtype=bool)
    if n > 1:
        dmins = (dt64[1:] - dt64[:-1]) / np.timedelta64(1, "m")
        gap_break[1:] = dmins > float(args.gap_minutes)

    # Prepare outputs
    z_all = np.full(n, np.nan, dtype=float)
    s_res_all = np.full(n, np.nan, dtype=float)
    B_all = np.full(n, np.nan, dtype=float)
    w_i_all = np.full(n, np.nan, dtype=float)
    delta_all = np.full(n, np.nan, dtype=float)

    w_max = _oddify(args.w_max)
    w_min = _oddify(args.w_min)
    if w_min > w_max:
        w_min = w_max

    # Process each segment
    seg_starts = [0]
    for i in range(1, n):
        if gap_break[i]:
            seg_starts.append(i)
    seg_starts.append(n)

    for si in range(len(seg_starts) - 1):
        s = seg_starts[si]
        e = seg_starts[si + 1]
        if e - s <= 0:
            continue

        y_seg = df["y"].iloc[s:e].to_numpy(dtype=float)
        dt_seg = df["datetime"].iloc[s:e].to_numpy(dtype="datetime64[ns]")

        if len(y_seg) < 3:
            # trivial segment
            z_all[s:e] = y_seg
            s_res_all[s:e] = 0.0
            B_all[s:e] = 0.0
            w_i_all[s:e] = float(w_min)
            delta_all[s:e] = _clip(0.0, args.delta_min, args.delta_max)
            continue

        z_seg, s_res_seg, B_seg, w_i_seg, delta_seg = _compute_segment_awat(
            dt=dt_seg,
            y=y_seg,
            w_max=w_max,
            w_min=w_min,
            delta_min=float(args.delta_min),
            delta_max=float(args.delta_max),
            poly_max_k=int(args.poly_max_k),
        )

        z_all[s:e] = z_seg
        s_res_all[s:e] = s_res_seg
        B_all[s:e] = B_seg
        w_i_all[s:e] = w_i_seg.astype(int)
        delta_all[s:e] = delta_seg

    noise_i = np.abs(df["y"].to_numpy(dtype=float) - z_all)

    out_df = pd.DataFrame({
        "datetime": df["datetime"],
        "y": df["y"].to_numpy(dtype=float),
        "z": z_all,
        "s_res": s_res_all,
        "B": B_all,
        "w_i": w_i_all.astype(int, copy=False),
        "delta_i": delta_all,
        "noise_i": noise_i,
    })

    # Save out CSV
    out_df.to_csv(out_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"[OK] Input: {in_path.resolve()}")
    print(f"[OK] Saved CSV: {out_path.resolve()}")
    print(f"[INFO] y column selected: {y_col}")

    # Noise intervals & plots
    noise_mask = np.isfinite(noise_i) & (noise_i > float(args.noise_eps))
    intervals = _find_intervals(
        dt=df["datetime"],
        mask=noise_mask,
        gap_minutes=float(args.gap_minutes),
        min_len=int(args.min_noise_len),
    )

    # Plot each interval
    saved = 0
    for idx, (l, r) in enumerate(intervals, start=1):
        t0 = df["datetime"].iloc[l]
        t1 = df["datetime"].iloc[r]
        fname = _sanitize_filename(f"noise_{idx:04d}_{_fmt_dt(t0)}__{_fmt_dt(t1)}.png")
        out_png = plots_dir / fname
        plot_noise_interval(
            out_path=out_png,
            dt=df["datetime"],
            y=out_df["y"].to_numpy(dtype=float),
            z=out_df["z"].to_numpy(dtype=float),
            interval_id=idx,
            l=l,
            r=r,
            max_xticks=8,
        )
        saved += 1

    print(f"[OK] Saved noise plots: {saved} files in {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
