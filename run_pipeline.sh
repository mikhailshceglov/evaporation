#!/bin/bash
set -e  # остановка при ошибке

echo "1) Убираем скачки (точно изменения извне)"
python3 topping_up/data_cleaner.py

echo "2.1) Детекция дождей"
python3 rain/apply_awat.py \
  --in data/cleaned_data.csv \
  --dt-col datetime \
  --y-col level_cleaned_mm \
  --noise-eps 0.086 \
  --min-noise-len 15 \
  --gap-minutes 10

echo "2.3) Убираем дождь"
python3 rain/remove_rain.py \
  --in rain/out/out.csv \
  --out rain/out/evap_only.csv \
  --out-html rain/out/evap_only.html \
  --dt-col datetime \
  --y-col y

echo "3) Сглаживание с помощью фильтра Калмана"
python3 Kalman_filter/filter.py

echo "3b) Создаём интерактивный график результата фильтра Калмана"
python3 Kalman_filter/interactive_result.py

# Автооткрытие интерактивного графика в браузере (Linux)
xdg-open result/after_kalman_evap.html

echo "Готово. Интерактивный график фильтра Калмана открыт в браузере."
