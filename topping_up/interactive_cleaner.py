from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# НАСТРОЙКИ
INPUT_FILE = Path("data/cleaned_data.xlsx")  # исходный и очищенный
OUTPUT_HTML = Path("topping_up/cleaned_data_plot.html")  # <-- обернули в Path

# ЗАГРУЗКА ДАННЫХ
df = pd.read_excel(INPUT_FILE)

# Проверим столбцы
required_cols = ["datetime", "level_raw_mm", "level_cleaned_mm"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"В {INPUT_FILE} нет столбца '{col}'")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

hover_dt = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

# Нахождение точек скачков
jump_mask = (df["level_raw_mm"] != df["level_cleaned_mm"]).to_numpy(bool)
jump_idx = np.where(jump_mask)[0]

# Построение интерактивного графика
fig = go.Figure()

# Исходный ряд
fig.add_trace(go.Scatter(
    x=np.arange(len(df)),
    y=df["level_raw_mm"],
    mode="lines",
    name="Исходный ряд",
    line=dict(color="red"),
    customdata=np.stack([hover_dt], axis=1),
    hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>level_raw_mm=%{y:.3f}<extra></extra>"
))

# Очищенный ряд
fig.add_trace(go.Scatter(
    x=np.arange(len(df)),
    y=df["level_cleaned_mm"],
    mode="lines",
    name="Очищенный ряд",
    line=dict(color="black"),
    customdata=np.stack([hover_dt], axis=1),
    hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>level_cleaned_mm=%{y:.3f}<extra></extra>"
))

# Точки скачков
fig.add_trace(go.Scatter(
    x=jump_idx,
    y=df.loc[jump_idx, "level_raw_mm"],
    mode="markers",
    name="Точки скачка",
    marker=dict(color="blue", size=6, symbol="circle"),
    customdata=np.stack([hover_dt[jump_idx]], axis=1),
    hovertemplate="i=%{x}<br>t=%{customdata[0]}<br>level_raw_mm=%{y:.3f}<extra></extra>"
))

# Layout
fig.update_layout(
    title="Очистка данных: исходный vs очищенный ряд",
    xaxis_title="Индекс",
    yaxis_title="Уровень воды, мм",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h"),
)

# Создаём папку для html
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# Сохраняем график
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(f"[OK] сохранён интерактивный график: {OUTPUT_HTML}")
