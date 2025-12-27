#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

try:
    from scipy.stats import theilslopes
except Exception:
    theilslopes = None

import plotly.graph_objects as go


SCRIPT_DIR = Path(__file__).resolve().parent

# ====== CHANGED DEFAULTS: everything into SCRIPT_DIR/out/ ======
OUT_DIR = SCRIPT_DIR / "out"
DEFAULT_IN   = OUT_DIR / "out_cleaned.csv"
DEFAULT_OUT  = OUT_DIR / "evap_only.csv"
DEFAULT_HTML = OUT_DIR / "evap_only.html"
# =============================================================

# ---- gap definition (we KEEP gaps as breaks!) ----
GAP_MINUTES = 10.0

# ---- rain candidates ----
SHORT_WIN = 31
BASE_WIN  = 721
ANOM_EPS = 0.08
RAIN_MIN_LEN = 20
HOLE_MAX_PTS = 5
RISE_MIN_MM = 0.20
SLOPE_MIN_MM_PER_MIN = 0.001

# ---- gate (intervention + ringdown) ----
JUMP_EPS_MM = 0.35
JUMP_EPS_STRONG_MM = 1.0
RING_WIN = 11
RING_STD_EPS_MM = 0.08
RING_CHECK_MINUTES = 15.0
DEAD_MINUTES_BASE = 20.0
DEAD_MINUTES_EXTRA = 20.0

# ---- counterfactual + shift ----
FIT_HOURS_PRE  = 6.0
POST_HOURS_OFF = 6.0
CLAMP_SLOPE_MAX_MM_PER_MIN = 0.0  # evap should not grow

MIN_SHIFT_MM_GATE = 0.05
EVENT_TAIL_MINUTES = 45.0

# ---- final safety: fix residual upward steps ----
STEP_FIX_EPS_MM = 0.20       # detect remaining upward jumps bigger than this
STEP_FIX_WIN_MIN = 30.0      # window (minutes) to estimate step size around the jump
STEP_FIX_PASSES = 3          # run several passes just in case

# ---- NEW: fix upward steps at time-gap boundaries ----
GAP_STEP_FIX_EPS_MM = 0.20
GAP_STEP_FIX_WIN_MIN = 60.0  # around boundary (minutes), more stable than tiny window


# =========================
# Utils
# =========================
def _gap_breaks(dt: pd.Series, gap_minutes: float) -> np.ndarray:
    n = len(dt)
    br = np.zeros(n, dtype=bool)
    if n > 1:
        t = dt.to_numpy(dtype="datetime64[ns]")
        gaps = (t[1:] - t[:-1]) / np.timedelta64(1, "m")
        br[1:] = gaps > float(gap_minutes)
    return br


def _median_dt_seconds(dt: pd.Series, br: np.ndarray) -> float:
    if len(dt) < 2:
        return 60.0
    t = dt.to_numpy(dtype="datetime64[ns]")
    dsec = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    good = ~br[1:]
    dsec = dsec[good]
    dsec = dsec[np.isfinite(dsec)]
    if dsec.size == 0:
        return 60.0
    med = float(np.median(dsec))
    if not np.isfinite(med) or med <= 0:
        return 60.0
    return med


def _segments_from_mask(mask: np.ndarray, br: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    n = len(mask)
    segs = []
    s = None
    for i in range(n):
        if br[i]:
            if s is not None:
                e = i - 1
                if e - s + 1 >= min_len:
                    segs.append((s, e))
                s = None

        if mask[i]:
            if s is None:
                s = i
        else:
            if s is not None:
                e = i - 1
                if e - s + 1 >= min_len:
                    segs.append((s, e))
                s = None

    if s is not None:
        e = n - 1
        if e - s + 1 >= min_len:
            segs.append((s, e))
    return segs


def _merge_segments(
    segs: list[tuple[int, int]],
    max_gap_pts: int,
    br: np.ndarray,
    block: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]

    for l2, r2 in segs[1:]:
        l1, r1 = merged[-1]
        gap = l2 - r1 - 1
        if gap <= max_gap_pts:
            mid_l = r1 + 1
            ok = True
            if mid_l <= l2:
                if br[mid_l:l2 + 1].any():
                    ok = False
                if block is not None and block[mid_l:l2].any():
                    ok = False
            if ok:
                merged[-1][1] = max(r1, r2)
                continue
        merged.append([l2, r2])

    return [(int(a), int(b)) for a, b in merged]


def _mask_from_segments(n: int, segs: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for l, r in segs:
        m[l:r + 1] = True
    return m


def _contiguous_time_segments(br: np.ndarray) -> list[tuple[int, int]]:
    n = len(br)
    segs = []
    l = 0
    for i in range(n):
        if br[i] and i > l:
            segs.append((l, i - 1))
            l = i
    if l < n:
        segs.append((l, n - 1))
    return segs


def _robust_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return 0.0, float(y[0]) if y.size else 0.0

    if theilslopes is not None and x.size >= 8:
        res = theilslopes(y, x)
        return float(res[0]), float(res[1])

    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _dilate_mask_tail(mask: np.ndarray, br: np.ndarray, tail_pts: int) -> np.ndarray:
    if tail_pts <= 0:
        return mask.copy()
    out = mask.copy()
    for seg_l, seg_r in _contiguous_time_segments(br):
        segs = _segments_from_mask(mask[seg_l:seg_r + 1], br[seg_l:seg_r + 1], 1)
        for l, r in segs:
            L = seg_l + l
            R = seg_l + r
            R2 = min(seg_r, R + tail_pts)
            out[L:R2 + 1] = True
    return out


# =========================
# Gate + Rain detection
# =========================
def _build_gate_mask(dt: pd.Series, y_short: pd.Series, gap_minutes: float) -> np.ndarray:
    n = len(dt)
    br = _gap_breaks(dt, gap_minutes)
    dt_step_sec = _median_dt_seconds(dt, br)

    base_pts  = max(1, int(np.ceil((DEAD_MINUTES_BASE * 60.0) / dt_step_sec)))
    extra_pts = max(0, int(np.ceil((DEAD_MINUTES_EXTRA * 60.0) / dt_step_sec)))
    check_pts = max(1, int(np.ceil((RING_CHECK_MINUTES * 60.0) / dt_step_sec)))

    dy = y_short.diff()
    abs_dy = dy.abs()

    jump_strong = (abs_dy > float(JUMP_EPS_STRONG_MM)).fillna(False).to_numpy(bool)
    jump_weak   = (abs_dy > float(JUMP_EPS_MM)).fillna(False).to_numpy(bool)

    dy_std = dy.rolling(RING_WIN, min_periods=max(2, RING_WIN // 2)).std()

    gate = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if br[i]:
            i += 1
            continue

        if not (jump_weak[i] or jump_strong[i]):
            i += 1
            continue

        j_end = min(n - 1, i + check_pts)
        if i + 1 <= j_end:
            br_in = np.where(br[i + 1:j_end + 1])[0]
            if br_in.size > 0:
                j_end = (i + 1) + int(br_in[0]) - 1

        ringdown = False
        if j_end > i:
            mx = float(np.nanmax(dy_std.iloc[i + 1:j_end + 1].to_numpy(float)))
            ringdown = np.isfinite(mx) and (mx >= float(RING_STD_EPS_MM))

        intervention = bool(jump_strong[i]) or (bool(jump_weak[i]) and ringdown)
        if not intervention:
            i += 1
            continue

        dead_pts = base_pts + (extra_pts if ringdown else 0)
        end = min(n - 1, i + dead_pts)

        if i + 1 <= end:
            br_in2 = np.where(br[i + 1:end + 1])[0]
            if br_in2.size > 0:
                end = (i + 1) + int(br_in2[0]) - 1

        gate[i:end + 1] = True
        i = end + 1

    return gate


def detect_rain_mask(dt: pd.Series, y: pd.Series, gap_minutes: float) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    br = _gap_breaks(dt, gap_minutes)

    y_short = y.rolling(SHORT_WIN, center=True, min_periods=1).mean()
    y_base  = y_short.rolling(BASE_WIN, center=False, min_periods=1).median()
    anom = y_short - y_base

    cand0 = (anom.to_numpy(float) > float(ANOM_EPS)).astype(bool)
    segs0 = _segments_from_mask(cand0, br, 1)
    segs0 = _merge_segments(segs0, HOLE_MAX_PTS, br, block=None)
    cand = _mask_from_segments(len(y), segs0)

    gate = _build_gate_mask(dt, y_short, gap_minutes)
    cand2 = cand & (~gate)

    segs1 = _segments_from_mask(cand2, br, 1)
    segs1 = _merge_segments(segs1, HOLE_MAX_PTS, br, block=gate)

    ys = y_short.to_numpy(float)

    final = []
    for l, r in segs1:
        if r - l + 1 < int(RAIN_MIN_LEN):
            continue

        net = float(ys[r] - ys[l])
        if not np.isfinite(net) or net < float(RISE_MIN_MM):
            continue

        dur_min = float((dt.iloc[r] - dt.iloc[l]).total_seconds() / 60.0)
        if dur_min <= 0:
            continue

        avg_slope = net / dur_min
        if avg_slope < float(SLOPE_MIN_MM_PER_MIN):
            continue

        final.append((l, r))

    rain = _mask_from_segments(len(y), final)
    return rain, gate, anom


# =========================
# Fix residual upward steps inside segments
# =========================
def _fix_residual_upward_steps(
    y: np.ndarray,
    br: np.ndarray,
    eps_mm: float,
    win_pts: int,
    passes: int,
) -> tuple[np.ndarray, int]:
    y = y.copy()
    fixed_count = 0

    for _ in range(max(1, passes)):
        changed = 0
        for seg_l, seg_r in _contiguous_time_segments(br):
            if seg_r - seg_l < 3:
                continue
            dy = np.diff(y[seg_l:seg_r + 1])
            cand = np.where(dy > float(eps_mm))[0]
            if cand.size == 0:
                continue

            for c in cand:
                i = seg_l + c + 1

                l1 = max(seg_l, i - win_pts)
                r1 = i - 1
                l2 = i
                r2 = min(seg_r, i + win_pts - 1)

                if r1 < l1 or r2 < l2:
                    continue

                before = y[l1:r1 + 1]
                after  = y[l2:r2 + 1]
                if before.size < 1 or after.size < 1:
                    continue

                med_before = float(np.median(before))
                med_after  = float(np.median(after))
                step = med_after - med_before

                if step > float(eps_mm):
                    y[i:seg_r + 1] -= step
                    changed += 1
                    fixed_count += 1
                    break

        if changed == 0:
            break

    return y, fixed_count


# =========================
# NEW: Fix upward steps across time-gap boundaries
# =========================
def _fix_upward_steps_across_gaps(
    y: np.ndarray,
    br: np.ndarray,
    eps_mm: float,
    win_pts: int,
) -> tuple[np.ndarray, int]:
    """
    For each gap boundary i where br[i]=True, compare medians before/after boundary.
    If the after-side is higher => shift the whole segment down by that step.
    We do NOT fill the gap; we only align vertical level across segments.
    """
    y = y.copy()
    n = len(y)
    fixed = 0

    segs = _contiguous_time_segments(br)
    starts = {l: (l, r) for (l, r) in segs}

    for i in np.where(br)[0]:
        if i <= 0 or i >= n:
            continue
        if i not in starts:
            continue

        rseg_l, rseg_r = starts[i]

        prev = None
        for (l, r) in segs:
            if r == i - 1:
                prev = (l, r)
                break
        if prev is None:
            continue
        lseg_l, lseg_r = prev

        b1 = max(lseg_l, (i - 1) - win_pts + 1)
        b2 = i - 1
        a1 = i
        a2 = min(rseg_r, i + win_pts - 1)

        before = y[b1:b2 + 1]
        after  = y[a1:a2 + 1]
        if before.size < 1 or after.size < 1:
            continue

        step = float(np.median(after) - np.median(before))
        if step > float(eps_mm):
            y[i:rseg_r + 1] -= step
            fixed += 1

    return y, fixed


# =========================
# Main removal: counterfactual + shift
# =========================
def remove_events_counterfactual(
    df: pd.DataFrame,
    dt_col: str,
    y_col: str,
    gap_minutes: float,
    fit_hours_pre: float,
    post_hours_off: float,
    clamp_slope_max: float,
    event_tail_minutes: float,
    min_shift_gate: float,
    use_gate: bool,
) -> pd.DataFrame:
    dt = df[dt_col]
    y_obs = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)

    rain_mask, gate_mask, anom = detect_rain_mask(dt, pd.Series(y_obs), gap_minutes)
    if not use_gate:
        gate_mask[:] = False

    br = _gap_breaks(dt, gap_minutes)
    dt_step_sec = _median_dt_seconds(dt, br)

    tail_pts = int(np.ceil((event_tail_minutes * 60.0) / dt_step_sec))
    event_mask = (rain_mask | gate_mask)
    event_mask = _dilate_mask_tail(event_mask, br, tail_pts=tail_pts)

    t0 = dt.iloc[0]
    t_min = (dt - t0).dt.total_seconds().to_numpy(float) / 60.0

    fit_pts  = max(8, int(np.ceil((fit_hours_pre * 3600.0) / dt_step_sec)))
    post_pts = max(8, int(np.ceil((post_hours_off * 3600.0) / dt_step_sec)))

    y_final = np.full_like(y_obs, np.nan, dtype=float)
    offset_cum = np.zeros_like(y_obs, dtype=float)
    slope_used = np.full_like(y_obs, np.nan, dtype=float)
    off_used = np.full_like(y_obs, np.nan, dtype=float)

    cum_global = 0.0
    for seg_l, seg_r in _contiguous_time_segments(br):
        cum = cum_global
        p = seg_l

        seg_events = _segments_from_mask(event_mask[seg_l:seg_r + 1], br[seg_l:seg_r + 1], 1)
        seg_events = [(seg_l + l, seg_l + r) for (l, r) in seg_events]

        for (l, r) in seg_events:
            if p <= l - 1:
                y_final[p:l] = y_obs[p:l] - cum
                offset_cum[p:l] = cum

            pre_end = l - 1
            pre_start = max(seg_l, pre_end - fit_pts + 1)
            pre_idx = np.arange(pre_start, pre_end + 1)
            pre_idx = pre_idx[(~event_mask[pre_idx]) & np.isfinite(y_obs[pre_idx])]

            if pre_idx.size < 8:
                pre_idx = np.arange(max(seg_l, l - 60), l)
                pre_idx = pre_idx[(~event_mask[pre_idx]) & np.isfinite(y_obs[pre_idx])]

            if pre_idx.size < 5:
                slope, intercept = 0.0, float((y_obs[pre_end] - cum) if (pre_end >= seg_l and np.isfinite(y_obs[pre_end])) else (y_obs[l] - cum))
            else:
                x_pre = t_min[pre_idx]
                y_pre = (y_obs[pre_idx] - cum)
                slope, intercept = _robust_slope_intercept(x_pre, y_pre)

            if np.isfinite(slope) and slope > clamp_slope_max:
                slope = clamp_slope_max

            if pre_end >= seg_l and np.isfinite(y_obs[pre_end]):
                t_anchor = float(t_min[pre_end])
                y_anchor = float((y_obs[pre_end] - cum))
            else:
                t_anchor = float(t_min[l])
                y_anchor = float((y_obs[l] - cum))
            intercept = y_anchor - slope * t_anchor

            x_e = t_min[l:r + 1]
            y_pred = slope * x_e + intercept
            y_final[l:r + 1] = y_pred
            offset_cum[l:r + 1] = cum
            slope_used[l:r + 1] = slope

            post_start = r + 1
            post_end = min(seg_r, r + post_pts)

            off = 0.0
            if post_start <= post_end:
                post_idx = np.arange(post_start, post_end + 1)
                post_idx = post_idx[(~event_mask[post_idx]) & np.isfinite(y_obs[post_idx])]
                if post_idx.size >= 8:
                    x_post = t_min[post_idx]
                    pred_post = slope * x_post + intercept
                    obs_post_adj = (y_obs[post_idx] - cum)
                    diff = obs_post_adj - pred_post
                    off = float(np.nanmedian(diff))
                    if not np.isfinite(off):
                        off = 0.0

            is_rain_like = bool(rain_mask[l:r + 1].mean() > 0.1)

            if is_rain_like:
                if off < 0:
                    off = 0.0
            else:
                if abs(off) < float(min_shift_gate):
                    off = 0.0

            cum += off
            off_used[l:r + 1] = off
            p = r + 1

        if p <= seg_r:
            y_final[p:seg_r + 1] = y_obs[p:seg_r + 1] - cum
            offset_cum[p:seg_r + 1] = cum

        cum_global = cum

    y_final = pd.Series(y_final).interpolate(limit_direction="both").to_numpy(float)

    out = df.copy()
    out["y_obs"] = y_obs
    out["y_final_raw"] = y_final

    out["rain_mask"] = rain_mask.astype(int)
    out["gate_mask"] = gate_mask.astype(int)
    out["event_mask"] = event_mask.astype(int)
    out["anom"] = np.asarray(anom, float)

    out["event_offset_cum"] = offset_cum
    out["event_slope_mm_per_min"] = slope_used
    out["event_offset_add_mm"] = off_used

    return out


# =========================
# HTML: KEEP gaps as breaks
# =========================
def write_final_html(df: pd.DataFrame, out_html: Path, gap_minutes: float) -> None:
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    y = pd.to_numeric(df["y_final"], errors="coerce")
    m = dt.notna() & y.notna()
    dt = dt[m].reset_index(drop=True)
    y = y[m].reset_index(drop=True)

    br = _gap_breaks(dt, gap_minutes)
    y_plot = y.to_numpy(float).copy()
    y_plot[br] = np.nan  # keep time gaps visible

    x = np.arange(len(y_plot), dtype=int)
    hover_dt = dt.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_plot, mode="lines",
        name="y_final (evap trend)",
        line=dict(color="black", width=1.4),
        customdata=np.stack([hover_dt], axis=1),
        hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y_final=%{y:.6f}<extra></extra>",
    ))

    title = f"Evaporation trend | {dt.iloc[0]} — {dt.iloc[-1]}"
    fig.update_layout(
        title=title,
        xaxis_title="Index (points)",
        yaxis_title="Level (mm)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out_csv", default=str(DEFAULT_OUT))
    ap.add_argument("--out-html", dest="out_html", default=str(DEFAULT_HTML))
    ap.add_argument("--no-html", action="store_true")

    ap.add_argument("--dt-col", default="datetime")
    ap.add_argument("--y-col", default="y")

    ap.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)

    ap.add_argument("--fit-hours-pre", type=float, default=FIT_HOURS_PRE)
    ap.add_argument("--post-hours-off", type=float, default=POST_HOURS_OFF)
    ap.add_argument("--clamp-slope-max", type=float, default=CLAMP_SLOPE_MAX_MM_PER_MIN)

    ap.add_argument("--event-tail-minutes", type=float, default=EVENT_TAIL_MINUTES)
    ap.add_argument("--min-shift-gate", type=float, default=MIN_SHIFT_MM_GATE)
    ap.add_argument("--no-gate", action="store_true")

    ap.add_argument("--step-fix-eps", type=float, default=STEP_FIX_EPS_MM)
    ap.add_argument("--step-fix-win-min", type=float, default=STEP_FIX_WIN_MIN)
    ap.add_argument("--step-fix-passes", type=int, default=STEP_FIX_PASSES)

    ap.add_argument("--gap-step-fix-eps", type=float, default=GAP_STEP_FIX_EPS_MM)
    ap.add_argument("--gap-step-fix-win-min", type=float, default=GAP_STEP_FIX_WIN_MIN)

    args = ap.parse_args()

    df = pd.read_csv(Path(args.in_csv), sep=";", decimal=",", encoding="utf-8-sig")

    if args.dt_col not in df.columns:
        raise ValueError(f"Нет dt-col '{args.dt_col}'. Есть: {list(df.columns)}")
    if args.y_col not in df.columns:
        if args.y_col == "y" and "y_obs" in df.columns:
            df["y"] = df["y_obs"]
        else:
            raise ValueError(f"Нет y-col '{args.y_col}'. Есть: {list(df.columns)}")

    df[args.dt_col] = pd.to_datetime(df[args.dt_col], errors="coerce")
    df = df.dropna(subset=[args.dt_col]).sort_values(args.dt_col).reset_index(drop=True)

    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df = df.dropna(subset=[args.y_col]).reset_index(drop=True)

    out = remove_events_counterfactual(
        df=df,
        dt_col=args.dt_col,
        y_col=args.y_col,
        gap_minutes=float(args.gap_minutes),
        fit_hours_pre=float(args.fit_hours_pre),
        post_hours_off=float(args.post_hours_off),
        clamp_slope_max=float(args.clamp_slope_max),
        event_tail_minutes=float(args.event_tail_minutes),
        min_shift_gate=float(args.min_shift_gate),
        use_gate=not args.no_gate,
    )

    out2 = out.copy()
    if args.dt_col != "datetime":
        out2["datetime"] = out2[args.dt_col]

    dt = pd.to_datetime(out2["datetime"], errors="coerce")
    br = _gap_breaks(dt, float(args.gap_minutes))
    dt_step_sec = _median_dt_seconds(dt, br)

    win_pts_in = max(8, int(np.ceil((float(args.step_fix_win_min) * 60.0) / dt_step_sec)))
    y1, fixed_in = _fix_residual_upward_steps(
        y=out2["y_final_raw"].to_numpy(float),
        br=br,
        eps_mm=float(args.step_fix_eps),
        win_pts=win_pts_in,
        passes=int(args.step_fix_passes),
    )

    win_pts_gap = max(8, int(np.ceil((float(args.gap_step_fix_win_min) * 60.0) / dt_step_sec)))
    y2, fixed_gap = _fix_upward_steps_across_gaps(
        y=y1,
        br=br,
        eps_mm=float(args.gap_step_fix_eps),
        win_pts=win_pts_gap,
    )

    out2["y_final"] = y2

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out2.to_csv(Path(args.out_csv), index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print(f"[OK] saved CSV: {args.out_csv}")
    print(f"[INFO] fixed steps inside segments: {fixed_in}")
    print(f"[INFO] fixed steps across gaps: {fixed_gap}")

    if not args.no_html:
        write_final_html(out2, Path(args.out_html), gap_minutes=float(args.gap_minutes))
        print(f"[OK] saved HTML: {args.out_html}")


if __name__ == "__main__":
    main()
