#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# SciPy optional
try:
    from scipy.stats import t as student_t  # type: ignore

    def t_crit_975(df: int) -> float:
        df = int(max(df, 1))
        return float(student_t.ppf(0.975, df))
except Exception:
    def t_crit_975(df: int) -> float:
        return 1.959963984540054


EPS = 1e-12


def make_odd(n: int) -> int:
    n = int(n)
    if n < 1:
        return 1
    return n if (n % 2 == 1) else (n + 1)


def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def parse_args(script_dir: Path) -> argparse.Namespace:
    """
    Defaults tied to project structure:
      project/
        data.csv
        rain/
          apply_awat.py
          out.csv
          noise_plots/
    """
    default_in = script_dir.parent / "data" / "data.csv"
    default_out = script_dir / "out.csv"
    default_plots = script_dir / "noise_plots"

    ap = argparse.ArgumentParser(description="Apply AWAT filter to time series in data.csv")
    ap.add_argument("--in", dest="in_path", default=str(default_in),
                    help="Input CSV (default: ../data.csv relative to script)")
    ap.add_argument("--out", dest="out_path", default=str(default_out),
                    help="Output CSV (default: rain/out.csv)")
    ap.add_argument("--plots-dir", dest="plots_dir", default=str(default_plots),
                    help="Dir for noise interval PNGs (default: rain/noise_plots)")
    ap.add_argument("--w-max", type=int, default=31, help="Maximum window width (odd number of points)")
    ap.add_argument("--w-min", type=int, default=5, help="Minimum window width (odd number of points)")
    ap.add_argument("--delta-min", type=float, default=0.081, help="Minimum threshold delta (mm)")
    ap.add_argument("--delta-max", type=float, default=0.24, help="Maximum threshold delta (mm)")
    ap.add_argument("--poly-max-k", type=int, default=6, help="Maximum polynomial degree for AICc search")
    ap.add_argument("--noise-eps", type=float, required=True, help="Noise threshold in mm for |y-z|")
    ap.add_argument("--min-noise-len", type=int, default=3, help="Min interval length (points) to save plot")
    ap.add_argument("--gap-minutes", type=float, default=10.0,
                    help="If time gap between consecutive points > this, split segments (minutes)")
    ap.add_argument("--y-col", type=str, default=None,
                    help="Optional: explicitly set y column name (otherwise auto-detect)")
    ap.add_argument("--max-xticks", type=int, default=10,
                    help="Max number of X tick labels on interval plots (picked from existing datapoints)")
    return ap.parse_args()


def resolve_path(p: str, script_dir: Path) -> Path:
    """
    If user passes relative path -> resolve relative to *current working dir*.
    If user passes something like ../data.csv while running from root, it still works.
    For defaults we already pass absolute-ish strings; still normalize here.
    """
    return Path(p).expanduser().resolve()


def choose_y_column(df: pd.DataFrame, dt_col: str, y_col_arg: Optional[str] = None) -> str:
    """
    Auto-pick y column.
    Priority: columns with 'уровень' (and 'мм' if present), then by numeric count.
    """
    if y_col_arg is not None:
        if y_col_arg not in df.columns:
            raise ValueError(f"--y-col '{y_col_arg}' not found in columns: {list(df.columns)}")
        return y_col_arg

    best_col = None
    best_score = -1

    for c in df.columns:
        if c == dt_col:
            continue

        s_num = pd.to_numeric(df[c], errors="coerce")
        count_num = int(s_num.notna().sum())
        if count_num == 0:
            continue

        name = str(c).strip().lower()
        score = count_num

        # strong preference for water level columns
        if "уров" in name or "уровень" in name:
            score += 1_000_000
        if "мм" in name:
            score += 200_000
        if "level" in name:
            score += 500_000
        if "water" in name:
            score += 200_000

        # slight penalty for obvious non-level columns
        if "температ" in name or "temp" in name:
            score -= 100_000
        if "напряж" in name or "акб" in name or "volt" in name:
            score -= 100_000

        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError("Could not auto-detect y column. Provide --y-col.")
    return str(best_col)


def split_by_gap_minutes(times: np.ndarray, gap_minutes: float) -> List[Tuple[int, int]]:
    n = len(times)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    gaps = (times[1:] - times[:-1]) / np.timedelta64(1, "m")
    breaks = np.where(gaps > gap_minutes)[0]
    segs = []
    start = 0
    for b in breaks:
        end = int(b)
        segs.append((start, end))
        start = int(b + 1)
    segs.append((start, n - 1))
    return segs


def window_bounds_centered(i: int, n: int, w: int) -> Tuple[int, int, int]:
    w = make_odd(int(w))
    if n <= w:
        l, r = 0, n - 1
        rlen = r - l + 1
        if rlen % 2 == 0 and rlen > 1:
            r -= 1
            rlen -= 1
        return l, r, rlen

    half = w // 2
    l = i - half
    r = i + half
    if l < 0:
        r += -l
        l = 0
    if r >= n:
        shift = r - (n - 1)
        l -= shift
        r = n - 1
        if l < 0:
            l = 0
    rlen = r - l + 1
    return int(l), int(r), int(rlen)


@dataclass
class FitStats:
    s_res: float
    s_dat: float
    B: float
    delta: float
    w_i: int


def polyfit_aicc_stats(
    times: np.ndarray,
    y: np.ndarray,
    i: int,
    w_max: int,
    w_min: int,
    delta_min: float,
    delta_max: float,
    k_max: int,
) -> FitStats:
    n = len(y)
    w_max = make_odd(w_max)
    w_min = make_odd(w_min)
    w_min = min(w_min, w_max)

    l, r, rlen = window_bounds_centered(i, n, w_max)
    ywin = y[l:r + 1]
    twin = times[l:r + 1]

    x = (twin - times[i]) / np.timedelta64(1, "m")
    x = x.astype(np.float64)

    if rlen < 3:
        s_res = 0.0
        ymean = float(np.mean(ywin)) if rlen > 0 else 0.0
        s_dat = float(np.sqrt(np.mean((ywin - ymean) ** 2))) if rlen > 0 else 0.0
        B = 1.0 if s_dat < EPS else float(min(max(s_res / s_dat, 0.0), 1.0))
        w_i = make_odd(max(w_min, int(round(B * w_max))))
        w_i = min(w_i, rlen if (rlen % 2 == 1) else max(rlen - 1, 1))
        delta = clip(s_res * t_crit_975(1), delta_min, delta_max)
        return FitStats(s_res=s_res, s_dat=s_dat, B=B, delta=delta, w_i=w_i)

    ymean = float(np.mean(ywin))
    s_dat = float(np.sqrt(np.mean((ywin - ymean) ** 2)))

    best_aicc = float("inf")
    best_k = 0
    best_ssq = None

    for k in range(0, int(k_max) + 1):
        nparams = k + 1
        if (rlen - nparams - 1) <= 0:
            continue
        try:
            coeff = np.polyfit(x, ywin, deg=k)
            yhat = np.polyval(coeff, x)
            resid = ywin - yhat
            ssq = float(np.sum(resid ** 2))
        except Exception:
            continue

        ssq_per = max(ssq / rlen, EPS)
        aicc = (rlen * np.log(ssq_per)) + (2.0 * nparams) + (2.0 * nparams * (nparams + 1) / (rlen - nparams - 1))
        if aicc < best_aicc:
            best_aicc = aicc
            best_k = k
            best_ssq = ssq

    if best_ssq is None:
        resid = ywin - ymean
        best_ssq = float(np.sum(resid ** 2))
        best_k = 0

    s_res = float(np.sqrt(best_ssq / rlen))

    if s_dat < EPS:
        B = 1.0
    else:
        B = float(s_res / s_dat)
    B = float(min(max(B, 0.0), 1.0))

    raw_w = max(float(w_min), float(B) * float(w_max))
    w_i = make_odd(int(round(raw_w)))
    w_i = min(w_i, w_max)
    max_odd_len = rlen if (rlen % 2 == 1) else max(rlen - 1, 1)
    w_i = min(w_i, max_odd_len)
    w_i = max(w_i, 1)

    df = int(max(rlen - (best_k + 1), 1))
    delta = clip(s_res * t_crit_975(df), delta_min, delta_max)

    return FitStats(s_res=s_res, s_dat=s_dat, B=B, delta=delta, w_i=w_i)


def moving_average_variable_window(y: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
    n = len(y)
    y = y.astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = make_odd(int(w_vec[i]))
        half = w // 2
        l = max(0, i - half)
        r = min(n - 1, i + half)
        out[i] = (csum[r + 1] - csum[l]) / float(r - l + 1)
    return out


def threshold_with_memory(tilde_y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    n = len(tilde_y)
    z = np.empty(n, dtype=np.float64)
    if n == 0:
        return z
    z[0] = float(tilde_y[0])
    for i in range(1, n):
        if abs(float(tilde_y[i]) - float(z[i - 1])) < float(delta[i]):
            z[i] = z[i - 1]
        else:
            z[i] = float(tilde_y[i])
    return z


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_tick_indices(n: int, max_ticks: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    max_ticks = max(2, int(max_ticks))
    if n <= max_ticks:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return np.unique(idx)


def plot_noise_interval(
    out_path: Path,
    dt: np.ndarray,  # datetime64[ns]
    y: np.ndarray,
    z: np.ndarray,
    interval_id: int,
    max_xticks: int,
) -> None:
    start_t = pd.to_datetime(dt[0]).to_pydatetime()
    end_t = pd.to_datetime(dt[-1]).to_pydatetime()

    n = len(dt)
    x = np.arange(n, dtype=int)  # x-axis = indices only

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, label="y (raw)")
    plt.plot(x, z, label="z (AWAT)")

    plt.title(f"Noise interval #{interval_id:03d}: {start_t} — {end_t}")
    plt.xlabel("Время (метки строго из данных)")
    plt.ylabel("y")

    # X ticks: ONLY from existing datapoints
    tick_idx = _pick_tick_indices(n, max_ticks=max_xticks)
    tick_labels = [pd.to_datetime(dt[i]).strftime("%Y-%m-%d %H:%M:%S") for i in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=30, ha="right")

    # "клетчатый" график
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.minorticks_on()

    if n <= 200:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        step = max(1, n // 50)
        ax.xaxis.set_minor_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    args = parse_args(script_dir)

    w_max = make_odd(args.w_max)
    w_min = make_odd(args.w_min)
    if w_min > w_max:
        w_min = w_max
    if args.delta_min > args.delta_max:
        raise ValueError("--delta-min must be <= --delta-max")

    in_path = resolve_path(args.in_path, script_dir)
    out_path = resolve_path(args.out_path, script_dir)
    plots_dir = resolve_path(args.plots_dir, script_dir)
    ensure_dir(plots_dir)
    ensure_dir(out_path.parent)

    df = pd.read_csv(in_path, sep=";", decimal=",", encoding="utf-8-sig")

    dt_col = "Дата/время"
    if dt_col not in df.columns:
        raise ValueError(f"Column '{dt_col}' not found. Columns: {list(df.columns)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["datetime"])

    y_col = choose_y_column(df, dt_col=dt_col, y_col_arg=args.y_col)
    df["y"] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["y"])

    df = df.sort_values("datetime").reset_index(drop=True)

    times = df["datetime"].to_numpy(dtype="datetime64[ns]")
    y_arr = df["y"].to_numpy(dtype=np.float64)
    n = len(df)
    if n == 0:
        raise ValueError("No valid rows after parsing datetime/y.")

    segments = split_by_gap_minutes(times, float(args.gap_minutes))

    s_res = np.full(n, np.nan, dtype=np.float64)
    B = np.full(n, np.nan, dtype=np.float64)
    w_i = np.full(n, 1, dtype=np.int32)
    delta_i = np.full(n, np.nan, dtype=np.float64)
    z = np.full(n, np.nan, dtype=np.float64)

    for (a, b) in segments:
        seg_len = b - a + 1
        if seg_len <= 0:
            continue

        t_seg = times[a:b + 1]
        y_seg = y_arr[a:b + 1]

        wmax_seg = min(w_max, seg_len if (seg_len % 2 == 1) else max(seg_len - 1, 1))
        wmin_seg = min(w_min, wmax_seg)

        if seg_len < 3 or wmax_seg < 3:
            s_res[a:b + 1] = 0.0
            B[a:b + 1] = 1.0
            w_i[a:b + 1] = 1
            delta_i[a:b + 1] = clip(0.0 * t_crit_975(1), args.delta_min, args.delta_max)
            z[a:b + 1] = y_seg
            continue

        w_vec = np.empty(seg_len, dtype=np.int32)
        delta_vec = np.empty(seg_len, dtype=np.float64)
        sres_vec = np.empty(seg_len, dtype=np.float64)
        B_vec = np.empty(seg_len, dtype=np.float64)

        for j in range(seg_len):
            stats = polyfit_aicc_stats(
                times=t_seg,
                y=y_seg,
                i=j,
                w_max=wmax_seg,
                w_min=wmin_seg,
                delta_min=float(args.delta_min),
                delta_max=float(args.delta_max),
                k_max=int(args.poly_max_k),
            )
            sres_vec[j] = stats.s_res
            B_vec[j] = stats.B
            delta_vec[j] = stats.delta
            w_vec[j] = stats.w_i

        tilde = moving_average_variable_window(y_seg, w_vec)
        z_seg = threshold_with_memory(tilde, delta_vec)

        s_res[a:b + 1] = sres_vec
        B[a:b + 1] = B_vec
        w_i[a:b + 1] = w_vec
        delta_i[a:b + 1] = delta_vec
        z[a:b + 1] = z_seg

    noise_i = np.abs(y_arr - z)

    out_df = pd.DataFrame({
        "datetime": pd.to_datetime(times),
        "y": y_arr,
        "z": z,
        "s_res": s_res,
        "B": B,
        "w_i": w_i.astype(int),
        "delta_i": delta_i,
        "noise_i": noise_i,
    })
    out_df.to_csv(out_path, sep=";", decimal=",", encoding="utf-8-sig", index=False)

    # intervals: respect gap breaks
    gap_break = np.zeros(n, dtype=bool)
    if n > 1:
        gaps = (times[1:] - times[:-1]) / np.timedelta64(1, "m")
        gap_break[1:] = gaps > float(args.gap_minutes)

    mask = noise_i > float(args.noise_eps)

    intervals: List[Tuple[int, int]] = []
    start = None
    for i in range(n):
        if gap_break[i]:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
    if start is not None:
        intervals.append((start, n - 1))

    saved = 0
    for idx, (a, b) in enumerate(intervals, start=1):
        if (b - a + 1) < int(args.min_noise_len):
            continue

        dt_seg = times[a:b + 1]
        y_seg = y_arr[a:b + 1]
        z_seg = z[a:b + 1]

        start_t = pd.to_datetime(dt_seg[0]).to_pydatetime()
        end_t = pd.to_datetime(dt_seg[-1]).to_pydatetime()
        fname = f"noise_{idx:03d}_{start_t:%Y%m%d_%H%M%S}__{end_t:%Y%m%d_%H%M%S}.png"
        plot_path = plots_dir / fname

        plot_noise_interval(
            plot_path,
            dt_seg,
            y_seg,
            z_seg,
            interval_id=idx,
            max_xticks=int(args.max_xticks),
        )
        saved += 1

    print(f"[OK] Input: {in_path}")
    print(f"[OK] Saved CSV: {out_path}")
    print(f"[OK] Saved noise plots: {saved} files in {plots_dir}/")
    print(f"[INFO] y column selected: {y_col}")


if __name__ == "__main__":
    main()
