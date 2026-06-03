# ML Fraud Detection Service

Сервис для автоматического обнаружения мошеннических транзакций в режиме батчевого скоринга. Обрабатывает CSV-файлы формата `test.csv` из соревнования [TETA ML 1 2025](https://www.kaggle.com/competitions/teta-ml-1-2025) с использованием предобученной CatBoost модели.

## Архитектура 

```
├── Dockerfile
├── README.md
├── requirements.txt
├── app/
│   └── app.py              # Ядро сервиса (watchdog + пайплайн)
├── models/
│   └── fraud_model.cbm     # Обученная CatBoost модель
├── src/
│   ├── preprocessing.py    # Препроцессинг данных
│   └── scorer.py           # Инференс модели
└── train_data/
    └── train.csv           # Обучающая выборка (для кодирования)
```

### Модель
- CatBoost (F1 ≈ 0.974 на валидации)
- Порог классификации: 0.5


### Как запустить проект

1. Взять `train.csv` из [соревнования](https://www.kaggle.com/competitions/teta-ml-1-2025) и поместить в `./train_data/`

2. Собрать образ:
```bash
docker build -t fraud_detector .
```

3. Запустить контейнер:
```bash
docker run -it --rm \
    -v ./input:/app/input \
    -v ./output:/app/output \
    fraud_detector
```

4. После появления в логах `File observer started` поместить файл `test.csv` в директорию `./input/`

5. Результаты сохраняются в `./output/`:
   - `sample_submission_*.csv`
   - `feature_importance.json`
   - `score_distribution.png`

Пример логов корректной работы сервиса: 

2026-06-03 03:45:14,895 - __main__ - INFO - Starting ML scoring service...

2026-06-03 03:45:14,895 - __main__ - INFO - Initializing ProcessingService...

2026-06-03 03:45:14,896 - preprocessing - INFO - Loading training data...

2026-06-03 03:45:16,156 - preprocessing - INFO - Raw train data imported. Shape: (786431, 14)

2026-06-03 03:45:24,230 - preprocessing - INFO - Train data processed. Shape: (786431, 21)

2026-06-03 03:45:24,230 - __main__ - INFO - Service initialized

2026-06-03 03:45:24,231 - __main__ - INFO - File observer started

2026-06-03 03:46:01,100 - __main__ - INFO - Processing file: /app/input/test.csv

2026-06-03 03:46:01,486 - __main__ - INFO - Starting preprocessing

2026-06-03 03:46:02,071 - preprocessing - INFO - Categorical merging completed. Shape: (262144, 13)

2026-06-03 03:46:02,149 - preprocessing - INFO - Time features added. Shape: (262144, 17)

2026-06-03 03:46:02,637 - preprocessing - INFO - Mean encoding completed. Shape: (262144, 28)

2026-06-03 03:46:04,929 - preprocessing - INFO - Preprocessing complete. Shape: (262144, 14)

2026-06-03 03:46:04,933 - __main__ - INFO - Making prediction

2026-06-03 03:46:05,329 - scorer - INFO - Prediction complete for file: /app/input/test.csv

2026-06-03 03:46:05,331 - scorer - INFO - Feature importance saved to: /app/output/feature_importance.json

2026-06-03 03:46:05,472 - matplotlib.font_manager - INFO - generated new fontManager

2026-06-03 03:46:05,639 - scorer - INFO - Score distribution chart saved to: /app/output/score_distribution.png

2026-06-03 03:46:05,755 - __main__ - INFO - Predictions saved to: sample_submission_20260603_034605.csv

^C 2026-06-03 03:46:56,394 - __main__ - INFO - Service stopped by user
