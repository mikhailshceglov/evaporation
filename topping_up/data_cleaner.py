import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# ПАРАМЕТРЫ
# =============================
THRESHOLD_MM = 1.5          # порог резкого изменения (мм)
DAYS_PER_MONTH_PLOT = 3     # сколько первых дней месяца показывать

TIME_COL = 'Дата/время'
LEVEL_COL = 'Уровень воды, мм'

# =============================
# АРГУМЕНТ
# =============================
if len(sys.argv) < 2:
    raise ValueError("Запуск: python data_cleaner.py input.xlsx")

INPUT_FILE = sys.argv[1]

OUTPUT_DATA = 'cleaned_data.xlsx'
OUTPUT_PLOT = 'level_before_after.png'
PLOTS_DIR = 'plots'

os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================
# ЗАГРУЗКА ДАННЫХ
# =============================
df = pd.read_excel(INPUT_FILE)

# Преобразуем дату в datetime
df['datetime'] = pd.to_datetime(
    df[TIME_COL],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Преобразуем уровень воды в числовой формат
df['level_mm'] = (
    df[LEVEL_COL]
    .astype(str)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

# =============================
# РАЗНОСТИ (вычисление изменений)
# =============================
df['delta'] = df['level_mm'].diff()

# =============================
# КОРРЕКЦИЯ ПРИЛИВОВ / ОТЛИВОВ
# =============================
df['level_cleaned'] = df['level_mm'].copy()
used = set()

# Множество точек скачков
jump_points = set()
st_p = set()
mid_p = set()
end_p = set()

for i in range(1, len(df) - 1):
    if i in used:
        continue

    if abs(df.loc[i, 'delta']) <= THRESHOLD_MM:
        continue

    # ---- поиск интервала события ----
    start = i
    end = i
    mid_p.add(i)

    # расширение влево
    j = i - 1
    d_event = abs(df.loc[j, 'level_mm'] - df.loc[j - 1, 'level_mm'])
    d_prev = abs(df.loc[j - 1, 'level_mm'] - df.loc[j - 2, 'level_mm'])
    
    if d_event > 2 * d_prev:
        start = j
        st_p.add(j)

    # расширение вправо
    j = i + 1
    d_event = abs(df.loc[j, 'level_mm'] - df.loc[j - 1, 'level_mm'])
    d_next = abs(df.loc[j + 1, 'level_mm'] - df.loc[j, 'level_mm'])
    
    if d_event > 2 * d_next:
        end = j
        end_p.add(j)

    jump_points.add(start)
    jump_points.add(start - 1)

    # ---- ПОШАГОВАЯ КОРРЕКЦИЯ ----
    for k in range(start, end + 1):
        jump = df.loc[k, 'level_mm'] - df.loc[k - 1, 'level_mm']
        df.loc[k:, 'level_cleaned'] -= jump
        df.loc[k, 'level_cleaned'] = df.loc[k - 1, 'level_cleaned']
        used.add(k)
        #jump_points.add(k)  # Добавляем индексы точек скачка в множество
    df.loc[end:, 'level_cleaned'] -= jump
# =============================
# СОХРАНЕНИЕ EXCEL
# =============================
out = df[['datetime', 'level_mm', 'level_cleaned']]
out.columns = ['datetime', 'level_raw_mm', 'level_cleaned_mm']
out.to_excel(OUTPUT_DATA, index=False)

# =============================
# ОБЩИЙ ГРАФИК
# =============================
plt.figure(figsize=(14, 6))
plt.plot(df['datetime'], df['level_mm'], color='red', linewidth=1.0, label='Исходный ряд')
plt.plot(df['datetime'], df['level_cleaned'], color='black', linewidth=1.5, label='Скорректированный ряд')

# Добавим маркеры для точек скачка
plt.scatter(df.loc[list(jump_points), 'datetime'], df.loc[list(jump_points), 'level_mm'], 
            color='blue', marker='o', label='Точки скачка', zorder=5)

plt.xlabel('Дата и время')
plt.ylabel('Уровень воды, мм')
plt.title('Удаление приливов и отливов')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

# =============================
# ПОМЕСЯЧНЫЕ ГРАФИКИ
# =============================
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

for (year, month), group in df.groupby(['year', 'month']):
    start_date = group['datetime'].min().normalize()
    end_date = start_date + pd.Timedelta(days=DAYS_PER_MONTH_PLOT)

    sub = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
    if sub.empty:
        continue

    plt.figure(figsize=(14, 6))
    plt.plot(sub['datetime'], sub['level_mm'], color='red', linewidth=1.0, label='Исходный ряд')
    plt.plot(sub['datetime'], sub['level_cleaned'], color='black', linewidth=1.5, label='Скорректированный ряд')

    # Добавим маркеры для точек скачка на помесячных графиках
    jump_sub = sub[sub['datetime'].isin(df.loc[list(jump_points), 'datetime'])]
    st_s = sub[sub['datetime'].isin(df.loc[list(st_p), 'datetime'])]
    mid_s = sub[sub['datetime'].isin(df.loc[list(mid_p), 'datetime'])]
    end_s = sub[sub['datetime'].isin(df.loc[list(end_p), 'datetime'])]
    plt.scatter(st_s['datetime'], st_s['level_mm'], 
                color='blue', marker='o', label='Точки скачка', zorder=5)
    plt.scatter(st_s['datetime'], st_s['level_cleaned'], 
                color='blue', marker='o', label='Точки скачка', zorder=5)
    plt.scatter(mid_s['datetime'], mid_s['level_mm'], 
                color='yellow', marker='o', label='Точки скачка', zorder=5)
    plt.scatter(mid_s['datetime'], mid_s['level_cleaned'], 
                color='yellow', marker='o', label='Точки скачка', zorder=5)
    plt.scatter(end_s['datetime'], end_s['level_mm'], 
                color='green', marker='o', label='Точки скачка', zorder=5)
    plt.scatter(end_s['datetime'], end_s['level_cleaned'], 
                color='green', marker='o', label='Точки скачка', zorder=5)

    plt.xlabel('Дата и время')
    plt.ylabel('Уровень воды, мм')
    plt.title(f'Первые {DAYS_PER_MONTH_PLOT} сутки {month:02d}.{year}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = f'level_{year}_{month:02d}_first_days.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=300)
    plt.close()

# =============================
# ГОТОВО
# =============================
print("Готово:")
print(f"Excel: {OUTPUT_DATA}")
print(f"Общий график: {OUTPUT_PLOT}")
print(f"Помесячные графики: {PLOTS_DIR}/")
