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
