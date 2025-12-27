#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================
# НАСТРОЙКИ ВРУЧНУЮ (КОНСТАНТЫ)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

START = None
END   = None

Y_MIN = None
Y_MAX = None

# разрывы данных (не склеивать через разрыв)
GAP_MINUTES = 10.0

# --- если в файле НЕТ rain_mask/gate_mask, считаем детекцию сами ---
SHORT_WIN = 31
BASE_WIN  = 721
ANOM_EPS = 0.08
RAIN_MIN_LEN = 20
HOLE_MAX_PTS = 5
RISE_MIN_MM = 0.20
SLOPE_MIN_MM_PER_MIN = 0.001

# gate
JUMP_EPS_MM = 0.35
JUMP_EPS_STRONG_MM = 1.0
RING_WIN = 11
RING_STD_EPS_MM = 0.08
RING_CHECK_MINUTES = 15.0
DEAD_MINUTES_BASE = 20.0
DEAD_MINUTES_EXTRA = 20.0

# --- ВИЗУАЛ ---
RAIN_RECT_ALPHA = 0.12
GATE_RECT_ALPHA = 0.10
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


def _build_gate_mask(dt: pd.Series, y_short: pd.Series) -> np.ndarray:
    n = len(dt)
    br = _gap_breaks(dt, GAP_MINUTES)
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


def _compute_rain_gate_from_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    dt = df["datetime"]
    y = df["y"]

    br = _gap_breaks(dt, GAP_MINUTES)

    y_short = y.rolling(SHORT_WIN, center=True, min_periods=1).mean()
    y_base  = y_short.rolling(BASE_WIN, center=False, min_periods=1).median()
    anom = y_short - y_base

    cand0 = (anom.to_numpy(float) > float(ANOM_EPS)).astype(bool)

    segs0 = _segments_from_mask(cand0, br, 1)
    segs0 = _merge_segments(segs0, HOLE_MAX_PTS, br, block=None)
    cand = _mask_from_segments(len(y), segs0)

    gate = _build_gate_mask(dt, y_short)

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


def main():
    ap = argparse.ArgumentParser()

    # ====== CHANGED DEFAULTS: everything into SCRIPT_DIR/out/ ======
    out_dir = SCRIPT_DIR / "out"
    ap.add_argument("--in", dest="in_csv", default=str(out_dir / "out.csv"),
                    help="CSV with columns datetime;y;z and optional y_evap/rain_mask/gate_mask (default: rain/out/out.csv)")
    ap.add_argument("--out-html", dest="out_html", default=str(out_dir / "interactive_plot.html"),
                    help="Output html (default: rain/out/interactive_plot.html)")
    # =============================================================

    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_html = Path(args.out_html)

    df = pd.read_csv(in_csv, sep=";", decimal=",", encoding="utf-8-sig")

    if "datetime" not in df.columns:
        raise ValueError(f"В {in_csv} нет колонки 'datetime'. Есть: {list(df.columns)}")
    if "y" not in df.columns:
        # allow older remove_rain outputs where y might be y_obs
        if "y_obs" in df.columns:
            df["y"] = df["y_obs"]
        else:
            raise ValueError(f"В {in_csv} нет колонки 'y' (и нет y_obs). Есть: {list(df.columns)}")

    if "z" not in df.columns:
        raise ValueError(f"В {in_csv} нет колонки 'z'. Есть: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df.dropna(subset=["y", "z"]).reset_index(drop=True)

    if "y_evap" in df.columns:
        df["y_evap"] = pd.to_numeric(df["y_evap"], errors="coerce")

    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()

    if START is not None:
        df = df[df["datetime"] >= pd.to_datetime(START)]
    if END is not None:
        df = df[df["datetime"] <= pd.to_datetime(END)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"После фильтра START/END не осталось данных. Файл: {dt_min} — {dt_max}")

    br = _gap_breaks(df["datetime"], GAP_MINUTES)

    # --- IMPORTANT: use rain_mask/gate_mask from file if present ---
    used_from_file = False
    if "rain_mask" in df.columns and "gate_mask" in df.columns:
        rain_mask = pd.to_numeric(df["rain_mask"], errors="coerce").fillna(0).to_numpy(int) != 0
        gate_mask = pd.to_numeric(df["gate_mask"], errors="coerce").fillna(0).to_numpy(int) != 0
        if "anom" in df.columns:
            anom = pd.to_numeric(df["anom"], errors="coerce").fillna(0.0)
        else:
            anom = pd.Series(np.zeros(len(df), dtype=float))
        used_from_file = True
    else:
        rain_mask, gate_mask, anom = _compute_rain_gate_from_y(df)

    rain_segs = _segments_from_mask(rain_mask, br, 1)
    gate_segs = _segments_from_mask(gate_mask, br, 1)

    # X = индекс (никаких придуманных времён на оси)
    x = np.arange(len(df), dtype=int)
    hover_dt = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    # quick diagnostics about y_evap
    has_evap = ("y_evap" in df.columns) and df["y_evap"].notna().any()
    if has_evap:
        diff = np.nanmax(np.abs(df["y_evap"].to_numpy(float) - df["y"].to_numpy(float)))
        print(f"[INFO] y_evap: OK, max|y_evap-y|={diff:.6f} mm")
    else:
        print("[INFO] y_evap: NOT FOUND in input file")

    print("[INFO] input:", in_csv)
    print("[INFO] used range:", df["datetime"].iloc[0], "—", df["datetime"].iloc[-1])
    print(f"[INFO] rain_segments={len(rain_segs)} gate_segments={len(gate_segs)} used_masks_from_file={used_from_file}")

    y = df["y"].to_numpy(float)
    z = df["z"].to_numpy(float)
    an = np.asarray(anom, float)

    fig = go.Figure()

    # rectangles
    shapes = []
    gate_fill = f"rgba(255,165,0,{GATE_RECT_ALPHA})"
    for l, r in gate_segs:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=l, x1=r, y0=0, y1=1,
                           fillcolor=gate_fill, line=dict(width=0), layer="below"))

    rain_fill = f"rgba(0,0,255,{RAIN_RECT_ALPHA})"
    for l, r in rain_segs:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=l, x1=r, y0=0, y1=1,
                           fillcolor=rain_fill, line=dict(width=0), layer="below"))

    if shapes:
        fig.update_layout(shapes=shapes)

    # legend proxies for rectangles
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="blue"),
                             name="rain segments"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="orange"),
                             name="gate segments"))

    # raw y
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        name="y (observed)",
        line=dict(color="black", width=1),
        customdata=np.stack([hover_dt, an], axis=1),
        hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y=%{y:.6f}<br>anom=%{customdata[1]:.6f}<extra></extra>",
    ))
    idx_y = len(fig.data) - 1

    # blue overlay for rain parts (raw y)
    idx_y_rain = []
    for (l, r) in rain_segs:
        seg_x = np.arange(l, r + 1, dtype=int)
        fig.add_trace(go.Scatter(
            x=seg_x, y=y[l:r + 1], mode="lines",
            name="y (rain overlay)",
            line=dict(color="blue", width=2),
            customdata=np.stack([hover_dt[l:r+1], an[l:r+1]], axis=1),
            hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y=%{y:.6f}<br>anom=%{customdata[1]:.6f}<extra></extra>",
            showlegend=False,
        ))
        idx_y_rain.append(len(fig.data) - 1)

    # z
    fig.add_trace(go.Scatter(
        x=x, y=z, mode="lines",
        name="z (AWAT)",
        line=dict(color="red", width=1.5),
        customdata=np.stack([hover_dt], axis=1),
        hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>z=%{y:.6f}<extra></extra>",
    ))
    idx_z = len(fig.data) - 1

    # y_evap
    idx_evap = None
    if has_evap:
        yev = df["y_evap"].to_numpy(float)
        fig.add_trace(go.Scatter(
            x=x, y=yev, mode="lines",
            name="y_evap (rain removed)",
            line=dict(color="green", width=1.5),
            customdata=np.stack([hover_dt], axis=1),
            hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y_evap=%{y:.6f}<extra></extra>",
        ))
        idx_evap = len(fig.data) - 1

    # layout
    title = (f"Interactive plot | {df['datetime'].iloc[0]} — {df['datetime'].iloc[-1]}"
             f" | rain_segments={len(rain_segs)} | gate_segments={len(gate_segs)}")
    fig.update_layout(
        title=title,
        xaxis_title="Index (points)",
        yaxis_title="Level (mm)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )

    if Y_MIN is not None or Y_MAX is not None:
        fig.update_yaxes(range=[Y_MIN, Y_MAX])

    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    # buttons
    if idx_evap is not None:
        n_tr = len(fig.data)

        def vis_raw():
            v = [True] * n_tr
            v[idx_evap] = False
            v[idx_y] = True
            for j in idx_y_rain:
                v[j] = True
            v[idx_z] = True
            return v

        def vis_evap():
            v = [True] * n_tr
            v[idx_evap] = True
            v[idx_y] = False
            for j in idx_y_rain:
                v[j] = False
            v[idx_z] = True
            return v

        def vis_all():
            v = [True] * n_tr
            v[idx_evap] = True
            v[idx_y] = True
            for j in idx_y_rain:
                v[j] = True
            v[idx_z] = True
            return v

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.01,
                    y=1.10,
                    buttons=[
                        dict(label="All", method="update", args=[{"visible": vis_all()}]),
                        dict(label="Raw y + z", method="update", args=[{"visible": vis_raw()}]),
                        dict(label="Evap-only y_evap + z", method="update", args=[{"visible": vis_evap()}]),
                    ],
                )
            ]
        )

        # default mode = All
        fig.update_traces(visible=True)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] saved: {out_html}")


if __name__ == "__main__":
    main()
