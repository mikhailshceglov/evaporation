import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# ПАРАМЕТРЫ ФИЛЬТРА КАЛМАНА
# =============================
Q_LEVEL = 1e-5          # шум уровня (мм^2)
R_LEVEL = 0.05           # шум измерения (мм)
EMA_ALPHA = 0.3          # скорость адаптации наклона
Q_TREND_MIN = 1e-6
Q_TREND_MAX = 5e-3
DAYS_PER_MONTH_PLOT = 3

# =============================
# ПУТИ
# =============================
INPUT_FILE = "rain/out/evap_only.csv"
OUTPUT_DATA = "result/after_kalman_evap.csv"
PLOTS_DIR = "Kalman_filter/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================
# ЗАГРУЗКА ДАННЫХ
# =============================
df = pd.read_csv(INPUT_FILE, sep=";", decimal=",", encoding="utf-8-sig")

if "datetime" not in df.columns:
    raise ValueError("В файле нет столбца 'datetime'")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

# =============================
# ВЫБОР СТОЛБЦА УРОВНЯ ВОДЫ
# =============================
PREFERRED_COLS = ["y_final", "evap_only", "level_cleaned_mm", "level_mm", "y"]
Y_COL = None
for col in PREFERRED_COLS:
    if col in df.columns:
        Y_COL = col
        break

if Y_COL is None:
    raise ValueError(f"Не найден столбец уровня воды. Доступные: {list(df.columns)}")

df = df.dropna(subset=[Y_COL]).reset_index(drop=True)
print(f"[INFO] Используется столбец уровня воды: {Y_COL}")

z = df[Y_COL].values
n = len(z)
if n < 2:
    raise ValueError("Недостаточно данных для фильтра Калмана")

# =============================
# dt — реальный шаг времени (в часах)
# =============================
dt = df["datetime"].diff().dt.total_seconds().fillna(60).values / 3600.0
dt[dt <= 0] = np.median(dt[dt > 0])

# =============================
# ИНИЦИАЛИЗАЦИЯ ФИЛЬТРА
# =============================
x = np.zeros((n, 2))  # состояние: [уровень, тренд]
P = np.eye(2)

x[0, 0] = z[0]
x[0, 1] = (z[1] - z[0]) / dt[1] if n > 1 else 0.0

H = np.array([[1.0, 0.0]])
R = np.array([[R_LEVEL]])
local_slope_ema = x[0, 1]

# =============================
# ФИЛЬТР КАЛМАНА (АДАПТИВНЫЙ)
# =============================
for k in range(1, n):
    slope_obs = (z[k] - z[k - 1]) / dt[k]
    local_slope_ema = EMA_ALPHA * slope_obs + (1 - EMA_ALPHA) * local_slope_ema
    Q_trend = np.clip(local_slope_ema ** 2, Q_TREND_MIN, Q_TREND_MAX)
    Q = np.array([[Q_LEVEL, 0.0], [0.0, Q_trend]])

    F = np.array([[1.0, dt[k]], [0.0, 1.0]])
    x_pred = F @ x[k - 1]
    P_pred = F @ P @ F.T + Q

    y = z[k] - (H @ x_pred)[0]
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x[k] = x_pred + (K.flatten() * y)
    P = (np.eye(2) - K @ H) @ P_pred

df["level_kalman"] = x[:, 0]
df["evap_rate_mm_per_h"] = x[:, 1]

# =============================
# СОХРАНЕНИЕ CSV И EXCEL
# =============================
df_out = df[["datetime", Y_COL, "level_kalman", "evap_rate_mm_per_h"]]
df_out.to_csv(OUTPUT_DATA, index=False, sep=";", decimal=",", encoding="utf-8-sig")
df_out.to_excel(OUTPUT_DATA.replace(".csv", ".xlsx"), index=False)

# =============================
# ГРАФИК ВСЕГО РЯДА
# =============================
plt.figure(figsize=(14, 6))
plt.plot(df["datetime"], df[Y_COL], color="gray", alpha=0.6, label="Исходные данные")
plt.plot(df["datetime"], df["level_kalman"], color="blue", linewidth=2, label="Калман (адаптивный)")
plt.xlabel("Дата и время")
plt.ylabel("Уровень воды, мм")
plt.title("Фильтр Калмана для испарения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kalman_full.png"), dpi=300)
plt.close()

# =============================
# ГРАФИКИ ПЕРВЫХ ДНЕЙ КАЖДОГО МЕСЯЦА
# =============================
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month

for (year, month), g in df.groupby(["year", "month"]):
    start = g["datetime"].min().normalize()
    end = start + pd.Timedelta(days=DAYS_PER_MONTH_PLOT)
    sub = df[(df["datetime"] >= start) & (df["datetime"] < end)]
    if sub.empty:
        continue

    plt.figure(figsize=(14, 6))
    plt.plot(sub["datetime"], sub[Y_COL], color="gray", alpha=0.6, label="Исходные данные")
    plt.plot(sub["datetime"], sub["level_kalman"], color="blue", linewidth=2, label="Калман")
    plt.xlabel("Дата и время")
    plt.ylabel("Уровень воды, мм")
    plt.title(f"{month:02d}.{year} — первые {DAYS_PER_MONTH_PLOT} суток")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"kalman_{year}_{month:02d}.png"), dpi=300)
    plt.close()

print(f"[OK] Используемый столбец: {Y_COL}")
print(f"[OK] CSV: {OUTPUT_DATA}")
print(f"[OK] Excel: {OUTPUT_DATA.replace('.csv','.xlsx')}")
print(f"[OK] Графики: {PLOTS_DIR}/")
