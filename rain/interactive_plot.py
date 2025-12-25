#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================
# НАСТРОЙКИ ВРУЧНУЮ (КОНСТАНТЫ)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# Основной файл: результат AWAT
IN_CSV = SCRIPT_DIR / "out.csv"          # rain/out.csv

# Если out.csv ещё нет — можно показать просто сырые данные из ../data.csv
FALLBACK_CSV = SCRIPT_DIR.parent / "data" / "data.csv"

OUT_HTML = SCRIPT_DIR / "interactive_plot.html"

# Диапазон по времени (поставь None, чтобы взять весь диапазон)
START = None  # например "2025-08-01 21:27:00"
END   = None  # например "2025-08-01 23:00:00"

# Если хочешь ограничить Y вручную (иначе авто)
Y_MIN = None
Y_MAX = None

# Какая колонка y в исходном data.csv (если fallback). Можно None — выберется сама.
Y_COL_FALLBACK = None  # например "Уровень воды, мм"
# =========================


def choose_y_column_auto(df: pd.DataFrame, dt_col: str) -> str:
    best_col, best_score = None, -1
    for c in df.columns:
        if c == dt_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        cnt = int(s.notna().sum())
        if cnt == 0:
            continue
        name = str(c).strip().lower()
        score = cnt
        if "уров" in name or "уровень" in name:
            score += 1_000_000
        if "мм" in name:
            score += 200_000
        if "температ" in name or "temp" in name:
            score -= 100_000
        if "напряж" in name or "акб" in name or "volt" in name:
            score -= 100_000
        if score > best_score:
            best_score, best_col = score, c
    if best_col is None:
        raise ValueError("Не смог выбрать y-колонку автоматически. Укажи Y_COL_FALLBACK.")
    return str(best_col)


def load_data():
    """
    Returns (df, mode) where mode in {"awat","raw"}.
    df has columns: datetime, y, (optional) z
    """
    if IN_CSV.exists():
        df = pd.read_csv(IN_CSV, sep=";", decimal=",", encoding="utf-8-sig")
        if "datetime" not in df.columns:
            raise ValueError("В rain/out.csv нет колонки 'datetime'.")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        # y/z могут быть строками из-за decimal=',', но мы читаем decimal=',' => будут числа
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["z"] = pd.to_numeric(df["z"], errors="coerce")
        df = df.dropna(subset=["y", "z"])
        return df[["datetime", "y", "z"]], "awat"

    # fallback: raw data.csv
    if not FALLBACK_CSV.exists():
        raise FileNotFoundError(f"Нет {IN_CSV} и нет {FALLBACK_CSV}")

    df = pd.read_csv(FALLBACK_CSV, sep=";", decimal=",", encoding="utf-8-sig")
    dt_col = "Дата/время"
    if dt_col not in df.columns:
        raise ValueError(f"В ../data.csv нет колонки '{dt_col}'.")
    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    y_col = Y_COL_FALLBACK or choose_y_column_auto(df, dt_col=dt_col)
    df["y"] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["y"])
    return df[["datetime", "y"]], "raw"


def main():
    df, mode = load_data()

    # Реальный диапазон в данных
    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()

    # Фильтр по START/END (константы)
    if START is not None:
        df = df[df["datetime"] >= pd.to_datetime(START)]
    if END is not None:
        df = df[df["datetime"] <= pd.to_datetime(END)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "После фильтра по START/END не осталось данных.\n"
            f"Доступный диапазон в файле: {dt_min} — {dt_max}\n"
            f"Твои START/END: {START} — {END}\n"
            "Поставь START/END внутрь этого диапазона или сделай их None."
        )

    # X = индекс точек (чтобы не было “придуманных” времен на оси)
    x = np.arange(len(df), dtype=int)
    hover_dt = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=df["y"],
        mode="lines",
        name="y (raw)" if mode == "raw" else "y (raw)",
        customdata=np.stack([hover_dt], axis=1),
        hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y=%{y:.6f}<extra></extra>",
    ))

    if mode == "awat" and "z" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["z"],
            mode="lines",
            name="z (AWAT)",
            customdata=np.stack([hover_dt], axis=1),
            hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>z=%{y:.6f}<extra></extra>",
        ))

    title = f"Interactive plot ({mode}) | {df['datetime'].iloc[0]} — {df['datetime'].iloc[-1]}"
    fig.update_layout(
        title=title,
        xaxis_title="Index (точки)",
        yaxis_title="Уровень / y",
        hovermode="x unified",
        template="plotly_white",
    )

    # Y limits if set
    if Y_MIN is not None or Y_MAX is not None:
        fig.update_yaxes(range=[Y_MIN, Y_MAX])

    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"[OK] saved: {OUT_HTML}")
    print(f"[INFO] data range in file: {dt_min} — {dt_max}")
    print(f"[INFO] used range: {df['datetime'].iloc[0]} — {df['datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
