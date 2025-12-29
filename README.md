Этот проект предназначен для обработки данных уровня воды: очистка от скачков, удаление дождевых событий и сглаживание с помощью адаптивного фильтра Калмана с возможностью интерактивной визуализации результатов.

---

# Структура проекта

- **`topping_up/`** — удаление резких скачков в исходном ряду.
- **`rain/`** — детекция и удаление дождевых событий.
- **`Kalman_filter/`** — сглаживание данных с помощью фильтра Калмана.
- **`data/`** — исходные и промежуточные данные (CSV и Excel).
- **`result/`** — результаты фильтра Калмана (Excel, CSV, HTML).
- **`plots/`** — графики и визуализации (по месяцам и общие).

---

## Установка зависимостей

Создайте виртуальное окружение и установите необходимые пакеты:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#Пошаговая инструкция по запуску
#Удаление скачков (резкие изменения извне)

Этап убирает внезапные изменения уровня воды, вызванные измерениями или внешними событиями.

## Запуск очистки данных
python3 topping_up/data_cleaner.py

## Интерактивная визуализация до и после очистки
python3 topping_up/interactive_cleaner.py

## Открыть интерактивный график в браузере
xdg-open topping_up/cleaned_data_plot.html


Результат: data/cleaned_data.xlsx и data/cleaned_data.csv с очищенными данными.

#Детекция и удаление дождевых событий
## Расчёт дождевых событий с AWAT
python3 rain/apply_awat.py \
  --in data/cleaned_data.csv \
  --dt-col datetime \
  --y-col level_cleaned_mm \
  --noise-eps 0.086 \
  --min-noise-len 15 \
  --gap-minutes 10


--in — входной CSV с очищенными данными.
--dt-col — имя столбца с датой и временем.
--y-col — имя столбца с уровнем воды.
--noise-eps — порог шумов.
--min-noise-len — минимальная длина шумового сегмента.
--gap-minutes — минимальный разрыв между сегментами, чтобы не объединять.

## Интерактивная визуализация дождевых сегментов
python3 rain/interactive_plot.py
xdg-open rain/out/interactive_plot.html

Синий — сегменты дождя
Оранжевый — сегменты шлюза (gate)

## Убираем дождь из данных
python3 rain/remove_rain.py \
  --in rain/out/out.csv \
  --out rain/out/evap_only.csv \
  --out-html rain/out/evap_only.html \
  --dt-col datetime \
  --y-col y

Результат: rain/out/evap_only.csv и HTML интерактивная визуализация rain/out/evap_only.html.

## Просмотр очищенного от дождя ряда
xdg-open rain/out/evap_only.html

#Сглаживание с помощью фильтра Калмана

Фильтр сглаживает данные, сохраняя тренд, используя адаптивный подход с EMA для локального наклона.

## Применение фильтра Калмана
python3 Kalman_filter/filter.py

## Интерактивная визуализация результата фильтра
python3 Kalman_filter/interactive_result.py

## Открыть результат в браузере
xdg-open result/after_kalman_evap.html


#Результат:
result/after_kalman_evap.xlsx — Excel с исходным, очищенным и сглаженным рядом.
result/after_kalman_evap.csv — CSV с аналогичными данными.
result/after_kalman_evap.html — интерактивная визуализация.

#Альтернатива: автоматический запуск всех этапов

Вместо ручного выполнения всех команд можно использовать скрипт run_pipeline.sh, который выполняет все шаги (кроме интерактивных графиков для промежуточных этапов) и открывает итоговый HTML с фильтром Калмана:

bash run_pipeline.sh
