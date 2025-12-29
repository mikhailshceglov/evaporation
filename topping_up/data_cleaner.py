import os
import pandas as pd
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ
THRESHOLD_MM = 0.7
DAYS_PER_MONTH_PLOT = 3

TIME_COL = "Дата/время"
LEVEL_COL = "Уровень воды, мм"

# ПУТИ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLOTS_DIR = os.path.join(PROJECT_DIR, "topping_up/plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_XLSX = os.path.join(DATA_DIR, "data.xlsx")
INPUT_CSV  = os.path.join(DATA_DIR, "data.csv")

OUTPUT_XLSX = os.path.join(DATA_DIR, "cleaned_data.xlsx")
OUTPUT_CSV  = os.path.join(DATA_DIR, "cleaned_data.csv")
OUTPUT_PLOT = os.path.join(PLOTS_DIR, "level_before_after.png")

# ЗАГРУЗКА ДАННЫХ
if os.path.exists(INPUT_XLSX):
    df = pd.read_excel(INPUT_XLSX)
elif os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV, sep=";", decimal=",", encoding="utf-8-sig")
else:
    raise FileNotFoundError("В папке data/ не найден data.xlsx или data.csv")

df["datetime"] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

df["level_mm"] = (
    df[LEVEL_COL]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

df["delta"] = df["level_mm"].diff()

# КОРРЕКЦИЯ СКАЧКОВ 
df["level_cleaned"] = df["level_mm"].copy()
used = set()

jump_points = set()
st_p, mid_p, end_p = set(), set(), set()

for i in range(2, len(df) - 2):

    if i in used:
        continue
    if abs(df.loc[i, "delta"]) <= THRESHOLD_MM:
        continue

    start = i
    end = i
    mid_p.add(i)

    j = i - 1
    d_event = abs(df.loc[j, "level_mm"] - df.loc[j - 1, "level_mm"])
    d_prev  = abs(df.loc[j - 1, "level_mm"] - df.loc[j - 2, "level_mm"])
    if d_event > 2 * d_prev and j not in used:
        start = j
    st_p.add(start)

    j = i + 1
    d_event = abs(df.loc[j, "level_mm"] - df.loc[j - 1, "level_mm"])
    d_next  = abs(df.loc[j + 1, "level_mm"] - df.loc[j, "level_mm"])
    if d_event > 2 * d_next:
        end = j
        end_p.add(j)

    jump_points.update([start, start - 1])

    for k in range(start, end + 1):
        jump = df.loc[k, "level_mm"] - df.loc[k - 1, "level_mm"]
        df.loc[k:, "level_cleaned"] -= jump
        used.add(k)

# СОХРАНЕНИЕ ДАННЫХ
out = df[["datetime", "level_mm", "level_cleaned"]].copy()
out.columns = ["datetime", "level_raw_mm", "level_cleaned_mm"]

out.to_excel(OUTPUT_XLSX, index=False)
out.to_csv(OUTPUT_CSV, index=False, sep=";", decimal=",", encoding="utf-8-sig")


# ОБЩИЙ ГРАФИК

plt.figure(figsize=(14, 6))
plt.plot(df["datetime"], df["level_mm"], color="red", label="Исходный ряд")
plt.plot(df["datetime"], df["level_cleaned"], color="black", label="Скорректированный")

plt.scatter(
    df.loc[list(jump_points), "datetime"],
    df.loc[list(jump_points), "level_mm"],
    color="blue",
    s=20,
    label="Точки скачка",
    zorder=5
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

# ПЕРВЫЕ ДНИ КАЖДОГО МЕСЯЦА
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month

for (year, month), _ in df.groupby(["year", "month"]):

    start = df[(df.year == year) & (df.month == month)]["datetime"].min().normalize()
    end = start + pd.Timedelta(days=DAYS_PER_MONTH_PLOT)

    sub = df[(df.datetime >= start) & (df.datetime < end)]
    if sub.empty:
        continue

    plt.figure(figsize=(14, 6))
    plt.plot(sub.datetime, sub.level_mm, color="red", label="Исходный")
    plt.plot(sub.datetime, sub.level_cleaned, color="black", label="Очищенный")

    plt.scatter(sub[sub.index.isin(st_p)].datetime,
                sub[sub.index.isin(st_p)].level_mm,
                color="blue", label="Старт", zorder=5)

    plt.scatter(sub[sub.index.isin(mid_p)].datetime,
                sub[sub.index.isin(mid_p)].level_mm,
                color="orange", label="Середина", zorder=5)

    plt.scatter(sub[sub.index.isin(end_p)].datetime,
                sub[sub.index.isin(end_p)].level_mm,
                color="green", label="Конец", zorder=5)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"level_{year}_{month:02d}_first_days.png"),
        dpi=300
    )
    plt.close()

print("Готово.")
print(f"Очищенные данные: {OUTPUT_XLSX}, {OUTPUT_CSV}")
print(f"Графики: {PLOTS_DIR}/")