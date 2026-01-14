from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "result" / "after_kalman_evap.xlsx"
OUT_DIR = PROJECT_ROOT / "calc_result"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка данных
df = pd.read_excel(INPUT_FILE)

required_cols = ["datetime", "level_kalman"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"В файле нет столбца '{col}'")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Расчёт элементарного испарения
df["dlevel"] = df["level_kalman"].diff()
df["evap_step_mm"] = (-df["dlevel"])

# первая строка не имеет разности
df = df.dropna(subset=["evap_step_mm"]).reset_index(drop=True)

# Часовые интенсивности
df["hour"] = df["datetime"].dt.floor("h")

hourly = (
    df.groupby("hour", as_index=False)["evap_step_mm"]
    .sum()
    .rename(columns={
        "hour": "datetime",
        "evap_step_mm": "evap_mm"
    })
)

# Полусуточные величины
df["date"] = df["datetime"].dt.date
df["half_day"] = df["datetime"].dt.hour < 12
df["half_day"] = df["half_day"].map({True: "00–12", False: "12–24"})

half_daily = (
    df.groupby(["date", "half_day"], as_index=False)["evap_step_mm"]
    .sum()
    .rename(columns={"evap_step_mm": "evap_mm"})
)

# Суточные величины
daily = (
    df.groupby("date", as_index=False)["evap_step_mm"]
    .sum()
    .rename(columns={"evap_step_mm": "evap_mm"})
)

# Сохранение результатов
hourly.to_csv(OUT_DIR / "evap_hourly.csv", index=False)
hourly.to_excel(OUT_DIR / "evap_hourly.xlsx", index=False)

half_daily.to_csv(OUT_DIR / "evap_half_daily.csv", index=False)
half_daily.to_excel(OUT_DIR / "evap_half_daily.xlsx", index=False)

daily.to_csv(OUT_DIR / "evap_daily.csv", index=False)
daily.to_excel(OUT_DIR / "evap_daily.xlsx", index=False)

print(f"Результаты сохранены в папке: {OUT_DIR}")

# Создание графиков
def plot_daily(df_daily, out_dir):
    df_daily['month'] = pd.to_datetime(df_daily['date']).dt.month
    df_daily['day'] = pd.to_datetime(df_daily['date']).dt.day

    plt.figure(figsize=(12,6))
    for month, group in df_daily.groupby('month'):
        plt.plot(group['day'], group['evap_mm'], marker='o', label=f'Month {month}')
    plt.xlabel("День месяца")
    plt.ylabel("Суточное испарение (мм)")
    plt.title("Суточное испарение по дням месяца")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "evap_daily_plot.png")
    plt.close()


def plot_half_daily(df_half, out_dir):
    df_half['month'] = pd.to_datetime(df_half['date']).dt.month

    plt.figure(figsize=(14,6))
    for month, group in df_half.groupby('month'):
        group = group.sort_values(['date', 'half_day'])  # сортируем по дате и половине суток
        # создаём индекс полусуточек
        group = group.reset_index(drop=True)
        group['half_day_index'] = range(1, len(group)+1)
        plt.plot(group['half_day_index'], group['evap_mm'], marker='o', label=f'Month {month}')

    plt.xlabel("Полусуточные интервалы (каждая половина суток)")
    plt.ylabel("Испарение (мм)")
    plt.title("Полусуточное испарение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "evap_half_daily_plot.png")
    plt.close()


def plot_hourly(df_hourly, out_dir):
    df_hourly['month'] = pd.to_datetime(df_hourly['datetime']).dt.month
    df_hourly['hour'] = pd.to_datetime(df_hourly['datetime']).dt.hour

    plt.figure(figsize=(12,6))
    for month, group in df_hourly.groupby('month'):
        hourly_mean = group.groupby('hour')['evap_mm'].mean()
        plt.plot(hourly_mean.index, hourly_mean.values, marker='o', label=f'Month {month}')
    plt.xlabel("Час")
    plt.ylabel("Среднее испарение за час (мм)")
    plt.title("Часовое испарение (среднее по месяцам)")
    plt.xticks(range(0,24))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "evap_hourly_plot.png")
    plt.close()


# Вызов функций для построения графиков
plot_daily(daily, OUT_DIR)
plot_half_daily(half_daily, OUT_DIR)
plot_hourly(hourly, OUT_DIR)

print("Графики сохранены в папке:", OUT_DIR)