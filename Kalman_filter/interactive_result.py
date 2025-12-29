from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ПАРАМЕТРЫ
INPUT_XLSX = Path("result/after_kalman_evap.xlsx")
OUTPUT_HTML = Path("result/after_kalman_evap.html")

# ЗАГРУЗКА ДАННЫХ
df = pd.read_excel(INPUT_XLSX)

# Проверим нужные столбцы
required_cols = ["datetime", "y_final", "level_kalman"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"В {INPUT_XLSX} нет столбца '{col}'")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

hover_dt = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

# Построение интерактивного графика
fig = go.Figure()

# До фильтра
fig.add_trace(go.Scatter(
    x=np.arange(len(df)),
    y=df["y_final"],
    mode="lines",
    name="До фильтра (y_final)",
    line=dict(color="red"),
    customdata=np.stack([hover_dt], axis=1),
    hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>y_final=%{y:.3f}<extra></extra>"
))

# После фильтра
fig.add_trace(go.Scatter(
    x=np.arange(len(df)),
    y=df["level_kalman"],
    mode="lines",
    name="После фильтра (level_kalman)",
    line=dict(color="blue"),
    customdata=np.stack([hover_dt], axis=1),
    hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>level_kalman=%{y:.3f}<extra></extra>"
))

# Layout
fig.update_layout(
    title="Фильтр Калмана: до и после",
    xaxis_title="Индекс",
    yaxis_title="Уровень воды, мм",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h"),
)

# Создание папки, если её нет
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(f"[OK] сохранён интерактивный график: {OUTPUT_HTML}")
