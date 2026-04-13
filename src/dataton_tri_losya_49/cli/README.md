## CLI

В пакете `dataton_tri_losya_49.cli` собраны entrypoints командной строки.

Пакет намеренно тонкий: он **не** реализует бизнес-логику пайплайна.
CLI лишь:

- парсит аргументы;
- вызывает соответствующие функции/компоненты пайплайна;
- пишет артефакты (через `dataton_tri_losya_49.io`).

### infer.py

Основной пользовательский флоу: **CSV → submission.csv**.

Entry point:

```bash
uv run speakerid-infer --help
```

Сценарий:

1) читаем CSV со списком файлов (`--csv`, колонка `--filepath-col`);
2) детерминированно нарезаем/нормализуем длину аудио (`--chunk-seconds`);
3) извлекаем эмбеддинги ONNX-моделью (`--model`, `--providers`, `--output-name`);
4) строим retrieval (FAISS) и пишем submission (`--out`).

### experiment.py

Dev-флоу: запуск эксперимента по TOML-конфигу и сохранение артефактов.

Entry point:

```bash
uv run speakerid-experiment --help
```

Сценарий:

- `load_experiment_config()` читает TOML и возвращает `ExperimentConfig`.
- `run_experiment()` делает inference → index → (опционально) evaluation и сохраняет:
  - `config.toml` (копия конфига);
  - `embeddings.npz`;
  - `submission.csv`;
  - `metrics.json` (если доступны метки).

### Как расширять

- Если добавляете новый `encoder.type`/`index.backend`, CLI обычно менять не нужно:
  расширяйте фабрики в `pipeline/runner.py` и (при необходимости) схему в `pipeline/config.py`.
- Если добавляете новый сценарий запуска — создавайте новый модуль в `cli/` и
  регистрируйте entrypoint в `pyproject.toml`.
