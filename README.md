# Dataton_tri_losya_49

Система распознавания дикторов по голосу, устойчивая к различным искажениям.

Пайплайн: аудио -> эмбеддинги -> FAISS kNN -> submission CSV.

---

## Быстрый старт

### Локально (uv)

```bash
# Установить зависимости
uv sync

# Инференс: CSV -> submission.csv
uv run speakerid-infer --csv data/test.csv --out submission.csv --root data

# Эксперимент с метриками
uv run speakerid-experiment --config configs/experiments/onnx_baseline_test_public.toml
```

### Docker

```bash
# Скопировать шаблон переменных окружения и отредактировать
cp .env.example .env

# Собрать образ
docker compose build

# Инференс (параметры берутся из .env)
docker compose run --rm infer

# Инференс с переопределением прямо в командной строке
CSV_PATH=data/my_data.csv OUT_PATH=data/my_submission.csv docker compose run --rm infer

# Эксперимент (параметры берутся из .env)
docker compose run --rm experiment

# Эксперимент с другим конфигом
EXPERIMENT_CONFIG=configs/experiments/my_exp.toml docker compose run --rm experiment
```

**Требования для Docker:** Docker Engine ≥ 24, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/).

> `configs/`, `data/` и `models/` монтируются в контейнер — новые конфиги и датасеты доступны без пересборки образа.

---

## Переменные окружения (.env)

Docker Compose автоматически читает `.env` из корня проекта.
Шаблон со всеми доступными переменными: [`.env.example`](.env.example).

```bash
cp .env.example .env
# Отредактируйте .env под свои пути и конфиги
```

| Переменная | Описание |
|---|---|
| `CSV_PATH` | Входной CSV для `speakerid-infer` |
| `OUT_PATH` | Выходной submission CSV |
| `ROOT_PATH` | Корень для разрешения путей из CSV |
| `EXPERIMENT_CONFIG` | TOML-конфиг для `speakerid-experiment` |
| `BATCH_SIZE` | Размер батча энкодера |

Переменные из командной строки имеют приоритет над `.env`:
```bash
CSV_PATH=data/other.csv docker compose run --rm infer
```

---

## Структура проекта

```
.
├── configs/
│   ├── experiments/        # TOML-конфиги для dev-экспериментов (монтируются в Docker)
│   └── inference/          # TOML-конфиги для inference CLI
├── data/                   # Датасеты (не в git, монтируются в Docker)
├── experiments/            # Артефакты запусков: embeddings.npz, submission.csv, metrics.json
├── models/                 # Модели (не в git, монтируются в Docker)
├── notebooks/              # Jupyter-ноутбуки для анализа метрик
├── src/dataton_tri_losya_49/
│   ├── cli/                # CLI entrypoints (speakerid-infer, speakerid-experiment)
│   ├── io/                 # Сериализация артефактов (embeddings.npz, submission.csv)
│   └── pipeline/
│       ├── components/
│       │   ├── encoders/   # ONNX-энкодер и интерфейс для новых
│       │   ├── evaluators/ # Precision@K и интерфейс для новых
│       │   ├── indexers/   # FAISS IP-индекс и интерфейс для новых
│       │   └── loaders/    # SoundFile + CSV dataset loader
│       ├── config.py       # Парсинг TOML-конфигов
│       ├── interfaces.py   # Protocol-контракты компонентов
│       ├── registry.py     # Фабрики компонентов по типу из конфига
│       └── runner.py       # Оркестратор пайплайна
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Данные

CSV-файл должен содержать колонку `filepath` с относительными путями к аудиофайлам.
Колонка `speaker_id` опциональна — нужна только для вычисления метрик в `speakerid-experiment`.

```csv
filepath,speaker_id
audio/speaker_a_001.flac,spk_001
audio/speaker_a_002.flac,spk_001
audio/speaker_b_001.flac,spk_002
```

---

## Модели

Положите ONNX-модель в `models/`. Дефолтный путь: `models/baseline.onnx`.

Модель должна принимать батч waveforms `float32` формы `[B, T]` (16 kHz, mono)
и возвращать эмбеддинги `float32` формы `[B, D]` с именем выхода `embeddings`.

---

## Конфигурация

### Inference (`configs/inference/`)

Описывает компоненты и дефолты. Пути (`--csv`, `--out`, `--root`) передаются через CLI.
Подробнее: [`configs/inference/README.md`](configs/inference/README.md).

### Experiments (`configs/experiments/`)

Полный конфиг для dev-флоу: данные, энкодер, индекс, метрики.
Подробнее: [`configs/experiments/README.md`](configs/experiments/README.md).

---

## CLI

Подробная документация: [`src/dataton_tri_losya_49/cli/README.md`](src/dataton_tri_losya_49/cli/README.md).

### `speakerid-infer`

```
speakerid-infer --csv PATH --out PATH [--root DIR] [--model PATH]
                [--config PATH] [--k INT] [--batch-size INT] [--chunk-seconds FLOAT]
```

### `speakerid-experiment`

```
speakerid-experiment --config PATH [--batch-size INT]
```

Артефакты сохраняются в `experiments/<name>/`:
- `config.toml` — копия конфига
- `embeddings.npz` — матрица эмбеддингов
- `submission.csv` — результат kNN
- `metrics.json` — Precision@K (если есть `speaker_id`)

---

## Компоненты пайплайна

| Компонент | Реализация | Документация |
|-----------|-----------|--------------|
| Encoder | `OnnxEncoder` | [encoders/README.md](src/dataton_tri_losya_49/pipeline/components/encoders/README.md) |
| Indexer | `FaissInnerProductIndexer` | [indexers/README.md](src/dataton_tri_losya_49/pipeline/components/indexers/README.md) |
| Loader | `SoundFileWaveformLoader` + `CsvAudioDatasetLoader` | [loaders/README.md](src/dataton_tri_losya_49/pipeline/components/loaders/README.md) |
| Evaluator | `PrecisionAtKEvaluator` | [evaluators/README.md](src/dataton_tri_losya_49/pipeline/components/evaluators/README.md) |
| IO | `save_embeddings_npz`, `write_submission_csv` | [io/README.md](src/dataton_tri_losya_49/io/README.md) |

Все компоненты подключаются через `pipeline/registry.py` — для добавления нового энкодера/индексатора достаточно добавить ветку в registry и новый TOML-конфиг, не трогая runner или CLI.

---

## Ноутбуки

Для работы с [`notebooks/metrics_visual.ipynb`](notebooks/metrics_visual.ipynb) установите дополнительные зависимости:

```bash
uv sync --extra notebooks
```

---

## Разработка

```bash
# Установить с dev-зависимостями
uv sync --extra dev

# Линтер
uv run ruff check src/

# Форматирование
uv run ruff format src/
```

