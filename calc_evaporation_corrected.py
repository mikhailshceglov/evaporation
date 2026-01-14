from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ПУТИ
PROJECT_ROOT = Path(__file__).resolve().parent
LEVEL_FILE = PROJECT_ROOT / "result" / "after_kalman_evap.xlsx"
TEMP_FILE = PROJECT_ROOT / "data" / "data.xlsx"
OUT_DIR = PROJECT_ROOT / "calc_result_with_temperature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ПАРАМЕТРЫ
ALPHA = 0.21  # мм / °C — коэффициент термического расширения воды

# ЗАГРУЗКА
df_level = pd.read_excel(LEVEL_FILE)
df_temp = pd.read_excel(TEMP_FILE)

# переименование столбцов температуры
df_temp = df_temp.rename(columns={
    "Дата/время": "datetime",
    "Температура поверхностного слоя воды, град С": "temperature"
})

df_level["datetime"] = pd.to_datetime(df_level["datetime"])
df_temp["datetime"] = pd.to_datetime(df_temp["datetime"], dayfirst=True)

# объединение
df = pd.merge(
    df_level,
    df_temp[["datetime", "temperature"]],
    on="datetime",
    how="inner"
).sort_values("datetime").reset_index(drop=True)

# СТАРОЕ ИСПАРЕНИЕ
df["dlevel"] = df["level_kalman"].diff()
df["evap_step_mm"] = (-df["dlevel"])

# ПОДГОТОВКА
df["date"] = df["datetime"].dt.date
df["level_corrected"] = 0.0
df["evap_step_mm_corrected"] = 0.0

# ПРАВИЛЬНЫЙ ПЕРЕСЧЁТ ПО СУТКАМ
def recalc_day(day_df: pd.DataFrame) -> pd.DataFrame:
    day_df = day_df.copy()

    # первый замер суток — опорный
    h0 = day_df.iloc[0]["level_kalman"]
    T0 = day_df.iloc[0]["temperature"]

    # корректированный уровень ОТ ПЕРВОГО ЗАМЕРА
    day_df["level_corrected"] = (
        day_df["level_kalman"]
        - ALPHA * (day_df["temperature"] - T0)
    )

    # первое испарение не меняем
    day_df.iloc[0, day_df.columns.get_loc("evap_step_mm_corrected")] = (
        day_df.iloc[0]["evap_step_mm"]
    )

    # испарение по скорректированному уровню
    for i in range(1, len(day_df)):
        dlevel_corr = (
            day_df.iloc[i]["level_corrected"]
            - day_df.iloc[i - 1]["level_corrected"]
        )
        day_df.iloc[i, day_df.columns.get_loc("evap_step_mm_corrected")] = max(0, -dlevel_corr)

    return day_df

df_corrected = pd.concat(
    [recalc_day(g) for _, g in df.groupby("date")],
    ignore_index=True
)

# СОХРАНЕНИЕ
df_corrected.to_excel(
    OUT_DIR / "after_kalman_evap_temp_corrected.xlsx",
    index=False
)
df_corrected.to_csv(
    OUT_DIR / "after_kalman_evap_temp_corrected.csv",
    index=False
)

print("Файл с пересчитанным испарением сохранён")

# ГРАФИКИ (ПЕРВЫЕ 3 СУТОК)
first_days = sorted(df_corrected["date"].unique())[:3]

for day in first_days:
    day_df = df_corrected[df_corrected["date"] == day].copy()

    # ось X — часы суток
    hour = (
        day_df["datetime"].dt.hour
        + day_df["datetime"].dt.minute / 60
        + day_df["datetime"].dt.second / 3600
    )

    # кумулятивное испарение
    evap_old = day_df["evap_step_mm"].cumsum()
    evap_new = day_df["evap_step_mm_corrected"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(hour, evap_old, label="Старое испарение")
    plt.plot(hour, evap_new, label="С поправкой на температуру")
    plt.xlabel("Час суток")
    plt.ylabel("Накопленное испарение, мм")
    plt.title(f"Внутрисуточный ход испарения — {day}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"evap_diurnal_{day}.png")
    plt.close()

print("Графики для первых трёх суток построены")
