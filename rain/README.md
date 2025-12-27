# Rain pipeline (AWAT → разметка дождя → удаление дождя)

Входные **очищенные от отливов/доливов** данные лежат в:

* `data/cleaned_data.csv`

Все результаты складываются в папку:

* `rain/out/`

---

## 0) Один раз создать папку вывода

```bash
mkdir -p rain/out
```

---

## 1) AWAT + детекция дождей → CSV (и диагностические PNG по шуму)

Для `cleaned_data.csv` используем:

* `--dt-col datetime`
* `--y-col level_cleaned_mm`

```bash
python3 rain/apply_awat.py \
  --in data/cleaned_data.csv \
  --dt-col datetime \
  --y-col level_cleaned_mm \
  --noise-eps 0.086 \
  --min-noise-len 15 \
  --gap-minutes 10

```

Результаты:

* `rain/out/out_cleaned.csv` — AWAT-результат + разметка дождя/гейта (`rain_mask`, `gate_mask`)
* `rain/out/noise_plots_cleaned/` — диагностические графики интервалов шума (один интервал = один PNG)

---

## 2) Интерактивная визуализация результата AWAT

Рисует наблюдения `y` (чёрный) + сглаживание `z` (красный) и подсветку интервалов:

* дождь (`rain_mask`) — полупрозрачные прямоугольники
* гейт/вмешательство (`gate_mask`) — полупрозрачные прямоугольники

```bash
python3 rain/interactive_plot.py
xdg-open rain/out/interactive_plot.html

```

---

## 3) Удаление дождя → «только испарение» (аппроксимация + сдвиг вниз)

Скрипт:

* берёт `rain_mask`/`gate_mask` из входного CSV,
* внутри дождевых участков строит аппроксимацию трендом испарения,
* затем **сдвигает весь последующий ряд вниз**, чтобы убрать «ступеньки вверх».
* реальные разрывы по времени (gap > `--gap-minutes`) **оставляет разрывами**.

```bash
python3 rain/remove_rain.py \
  --in rain/out/out.csv \
  --out rain/out/evap_only.csv \
  --out-html rain/out/evap_only.html \
  --dt-col datetime \
  --y-col y
xdg-open rain/out/evap_only.html

```

Итоги:

* `rain/out/evap_only.csv` — финальный ряд `y_final` (испарение без дождя)
* `rain/out/evap_only.html` — интерактивный график итогового ряда

