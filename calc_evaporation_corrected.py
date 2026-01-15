from pathlib import Path
import pandas as pd
import numpy as np
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
    
    # НАСТРОЙКА ОСЕЙ
    # Ось X: деления каждый час (0, 1, 2, ..., 23)
    plt.xticks(range(0, 24, 1))
    
    # Ось Y: деления каждые 0.5 мм
    # Определяем диапазон данных
    all_y_values = np.concatenate([evap_old.values, evap_new.values])
    y_min = all_y_values.min()
    y_max = all_y_values.max()
    
    # Если данные отсутствуют или некорректны
    if np.isnan(y_min) or np.isnan(y_max):
        y_min = -0.5
        y_max = 0.5
    
    # Гарантируем минимальный диапазон
    if y_max - y_min < 0.5:
        y_center = (y_min + y_max) / 2
        y_min = y_center - 0.5
        y_max = y_center + 0.5
    
    # Создаем деления с шагом 0.5 мм
    y_start = np.floor(y_min * 2) / 2
    y_end = np.ceil(y_max * 2) / 2 + 0.5
    
    # Проверяем корректность диапазона
    if y_start >= y_end:
        y_start = -0.5
        y_end = 0.5
    
    # Создаем деления
    if y_end > y_start:  # Проверяем перед созданием arange
        y_ticks = np.arange(y_start, y_end, 0.5)
        plt.yticks(y_ticks)
    else:
        # Если что-то пошло не так, используем стандартные деления
        pass  # Оставляем стандартные деления matplotlib
    
    # Выделяем ноль (y=0)
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    
    # Подписываем ноль жирным
    ax = plt.gca()
    for tick in ax.yaxis.get_major_ticks():
        if tick.get_loc() == 0:
            tick.label1.set_fontweight('bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"evap_diurnal_{day}.png")
    plt.close()

print("Графики для первых трёх суток построены")
