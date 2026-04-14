## CLI

В пакете `dataton_tri_losya_49.cli` собраны entrypoints командной строки.

Пакет намеренно тонкий: он **не** реализует бизнес-логику пайплайна.
CLI лишь:

- парсит аргументы;
- загружает TOML-конфиг;
- вызывает соответствующие функции/компоненты пайплайна через registry;
- пишет артефакты (через `dataton_tri_losya_49.io`).

### infer.py

Основной пользовательский флоу: **CSV -> submission.csv**.

Entry point:

```bash
uv run speakerid-infer --help
```

Дизайн:
- **Компоненты и дефолты** (encoder/indexer/loader, chunk_seconds, topk и т.д.) задаются в inference TOML.
  Дефолтный конфиг: `configs/inference/baseline.toml`; переопределить: `--config`.
- **Пути** (`--csv`, `--out`, `--root`, `--model`) задаются аргументами CLI.
- **Providers** определяются автоматически (CUDA -> CPU); можно переопределить в TOML.

Минимальный вызов:

```bash
speakerid-infer --csv data/test_public.csv --out submission.csv
```

Дополнительные переопределения:

| Флаг | Что переопределяет |
|------|-------------------|
| `--model` | `encoder.model_path` из TOML |
| `--k` | `index.topk` из TOML |
| `--batch-size` | `defaults.batch_size` из TOML |
| `--chunk-seconds` | `defaults.chunk_seconds` из TOML |
| `--config` | путь до inference TOML |

### experiment.py

Dev-флоу: запуск эксперимента по TOML-конфигу и сохранение артефактов.

Entry point:

```bash
uv run speakerid-experiment --help
```

Сценарий:

- `load_experiment_config()` читает TOML и возвращает `ExperimentConfig`.
- `run_experiment()` делает inference -> index -> (опционально) evaluation и сохраняет:
  - `config.toml` (копия конфига);
  - `embeddings.npz`;
  - `submission.csv`;
  - `metrics.json` (если доступны метки).
- При запуске выводятся используемые компоненты (encoder type, indexer, loader, evaluator, providers).

### Как расширять

- Если добавляете новый `encoder.type` или `index.backend`:
  добавьте ветку в `pipeline/registry.py` - CLI менять не нужно.
- Если добавляете новый сценарий запуска - создавайте новый модуль в `cli/` и
  регистрируйте entrypoint в `pyproject.toml`.
