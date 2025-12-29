import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ
Q_level = 0.01       # процессный шум уровня
R = 0.5              # шум датчика
DAYS_PER_MONTH_PLOT = 3

# EMA (экспоненциальное скользящее среднее) параметры для локального наклона
EMA_ALPHA = 0.3      # скорость адаптации наклона
Q_trend_min = 1e-6
Q_trend_max = 0.05

INPUT_FILE = '../topping_up/cleaned_data.xlsx'
OUTPUT_DATA = 'after_kalman_data.xlsx'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ЗАГРУЗКА ДАННЫХ
df = pd.read_excel(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

z = df['level_cleaned_mm'].values
n = len(z)

# dt в часах
dt = df['datetime'].diff().dt.total_seconds().fillna(60).values / 3600

# ИНИЦИАЛИЗАЦИЯ
x = np.zeros((n,2))   # [level, trend]
P = np.eye(2)
x[0,0] = z[0]
x[0,1] = 0.0

H = np.array([[1,0]])
R_mat = np.array([[R]])

local_slope_ema_prev = 0.0

# АДАПТИВНЫЙ ФИЛЬТР С EMA
for k in range(1,n):
    # Вычисляем локальный наклон
    slope = (z[k] - z[k-1]) / dt[k]  # наклон в мм/час
    local_slope_ema = EMA_ALPHA * slope + (1 - EMA_ALPHA) * local_slope_ema_prev
    local_slope_ema_prev = local_slope_ema

    # Адаптивный Q_trend
    Q_trend = np.clip(abs(local_slope_ema)**2, Q_trend_min, Q_trend_max)
    Q = np.array([[Q_level,0],[0,Q_trend]])

    # Матрица перехода
    F = np.array([[1, dt[k]],[0,1]])

    # Prediction
    x_pred = F @ x[k-1]
    P_pred = F @ P @ F.T + Q

    # Update
    y = z[k] - (H @ x_pred)[0]
    S = H @ P_pred @ H.T + R_mat
# ПАРАМЕТРЫ ФИЛЬТРА КАЛМАНА

Q_LEVEL = 1e-5          # шум уровня (мм^2)
R_LEVEL = 0.05          # шум измерения (мм)

EMA_ALPHA = 0.3         # скорость адаптации наклона
Q_TREND_MIN = 1e-6
Q_TREND_MAX = 5e-3

DAYS_PER_MONTH_PLOT = 3

# ПУТИ
INPUT_FILE = "../rain/out/evap_only.csv"
OUTPUT_DATA = "after_kalman_evap.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ЗАГРУЗКА ДАННЫХ
df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

if "datetime" not in df.columns:
    raise ValueError("В файле нет столбца 'datetime'")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

# ВЫБОР СТОЛБЦА УРОВНЯ ВОДЫ
PREFERRED_COLS = [
    "y_final",
    "evap_only",
    "level_cleaned_mm",
    "level_mm",
    "y"
]

Y_COL = None
for col in PREFERRED_COLS:
    if col in df.columns:
        Y_COL = col
        break

if Y_COL is None:
    raise ValueError(
        f"Не найден столбец уровня воды. Доступные: {list(df.columns)}"
    )

print(f"[INFO] Используется столбец уровня воды: {Y_COL}")

df = df.dropna(subset=[Y_COL]).reset_index(drop=True)

z = df[Y_COL].values
n = len(z)

if n < 2:
    raise ValueError("Недостаточно данных для фильтра Калмана")

# dt — реальный шаг времени (в часах)
dt = (
    df["datetime"]
    .diff()
    .dt.total_seconds()
    .fillna(60)
    .values / 3600.0
)

dt[dt <= 0] = np.median(dt[dt > 0])

# ИНИЦИАЛИЗАЦИЯ ФИЛЬТРА
# состояние: [уровень, тренд]
x = np.zeros((n, 2))
P = np.eye(2)

x[0, 0] = z[0]
x[0, 1] = (z[1] - z[0]) / dt[1]

H = np.array([[1.0, 0.0]])
R = np.array([[R_LEVEL]])

local_slope_ema = x[0, 1]

# ФИЛЬТР КАЛМАНА (АДАПТИВНЫЙ)
for k in range(1, n):

    # --- наблюдаемый локальный наклон ---
    slope_obs = (z[k] - z[k - 1]) / dt[k]

    local_slope_ema = (
        EMA_ALPHA * slope_obs +
        (1.0 - EMA_ALPHA) * local_slope_ema
    )

    Q_trend = np.clip(
        local_slope_ema ** 2,
        Q_TREND_MIN,
        Q_TREND_MAX
    )

    Q = np.array([
        [Q_LEVEL, 0.0],
        [0.0, Q_trend]
    ])

    # --- матрица перехода ---
    F = np.array([
        [1.0, dt[k]],
        [0.0, 1.0]
    ])

    # --- prediction ---
    x_pred = F @ x[k - 1]
    P_pred = F @ P @ F.T + Q

    # --- update ---
    y = z[k] - (H @ x_pred)[0]
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x[k] = x_pred + (K.flatten() * y)
    P = (np.eye(2) - K @ H) @ P_pred

df['level_kalman_mm'] = x[:,0]

# СОХРАНЕНИЕ
df_out = df[['datetime','level_raw_mm','level_cleaned_mm','level_kalman_mm']]
df_out.to_excel(OUTPUT_DATA,index=False)

# ГРАФИК ВСЕГО РЯДА
plt.figure(figsize=(14,6))
plt.plot(df['datetime'], df['level_cleaned_mm'], color='gray', alpha=0.6, label='Очищенные')
plt.plot(df['datetime'], df['level_kalman_mm'], color='blue', linewidth=2, label='Калман адаптивный EMA')
plt.xlabel('Дата и время')
plt.ylabel('Уровень воды, мм')
plt.title('Фильтр Калмана')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,'kalman_full.png'), dpi=300)
plt.close()

# ПО ПЕРВЫМ ДНЯМ КАЖДОГО МЕСЯЦА
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

for (year, month), g in df.groupby(['year','month']):
    start = g['datetime'].min().normalize()
    end = start + pd.Timedelta(days=DAYS_PER_MONTH_PLOT)

    sub = df[(df['datetime']>=start) & (df['datetime']<end)]
    if sub.empty:
        continue

    plt.figure(figsize=(14,6))
    plt.plot(sub['datetime'], sub['level_cleaned_mm'], color='gray', alpha=0.6, label='Очищенные')
    plt.plot(sub['datetime'], sub['level_kalman_mm'], color='blue', linewidth=2, label='Калман адаптивный EMA')
    plt.xlabel('Дата и время')
    plt.ylabel('Уровень воды, мм')
    plt.title(f'{month:02d}.{year} первые {DAYS_PER_MONTH_PLOT} суток')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,f'kalman_{year}_{month:02d}.png'), dpi=300)
    plt.close()

print('данные и графики созданы.')
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
df["level_kalman"] = x[:, 0]
df["evap_rate_mm_per_h"] = x[:, 1]

df_out = df[
    ["datetime", Y_COL, "level_kalman", "evap_rate_mm_per_h"]
]

df_out.to_csv(
    OUTPUT_DATA,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

# ГРАФИК: ВЕСЬ РЯД
plt.figure(figsize=(14, 6))
plt.plot(
    df["datetime"],
    df[Y_COL],
    color="gray",
    alpha=0.6,
    label="Исходные данные"
)
plt.plot(
    df["datetime"],
    df["level_kalman"],
    color="blue",
    linewidth=2,
    label="Калман (адаптивный)"
)

plt.xlabel("Дата и время")
plt.ylabel("Уровень воды, мм")
plt.title("Фильтр Калмана для испарения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kalman_full.png"), dpi=300)
plt.close()

# ПЕРВЫЕ ДНИ КАЖДОГО МЕСЯЦА
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month

for (year, month), g in df.groupby(["year", "month"]):

    start = g["datetime"].min().normalize()
    end = start + pd.Timedelta(days=DAYS_PER_MONTH_PLOT)

    sub = df[(df["datetime"] >= start) & (df["datetime"] < end)]
    if sub.empty:
        continue

    plt.figure(figsize=(14, 6))
    plt.plot(
        sub["datetime"],
        sub[Y_COL],
        color="gray",
        alpha=0.6,
        label="Исходные"
    )
    plt.plot(
        sub["datetime"],
        sub["level_kalman"],
        color="blue",
        linewidth=2,
        label="Калман"
    )

    plt.xlabel("Дата и время")
    plt.ylabel("Уровень воды, мм")
    plt.title(f"{month:02d}.{year} — первые {DAYS_PER_MONTH_PLOT} суток")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"kalman_{year}_{month:02d}.png"),
        dpi=300
    )
    plt.close()

print(f"Используемый столбец: {Y_COL}")
print(f"CSV результат: {OUTPUT_DATA}")
print(f"Графики: {PLOTS_DIR}/")
