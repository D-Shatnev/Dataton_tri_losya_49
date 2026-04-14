# Конфиги инференса (TOML)

Конфиги инференса описывают **компоненты пайплайна** и их дефолтные параметры.  
Пути к данным (`--csv`, `--out`, `--root`) и опциональный путь к модели (`--model`) передаются через CLI — не хранятся в конфиге.

## Структура TOML

```toml
[encoder]
type        = "onnx"          # тип энкодера
model_path  = "models/..."    # путь к .onnx файлу (переопределяется через --model)
output_name = "embeddings"    # имя выходного тензора ONNX-модели
# providers = [...]           # ONNX Runtime providers; если не задано — авто-определение

[loader]
type      = "soundfile"       # тип загрузчика аудио
target_sr = 16000             # целевая частота дискретизации (Гц)
clip      = false             # обрезать ли waveform до chunk_seconds

[index]
backend = "faiss_ip"          # бэкенд индексатора (inner product / cosine)
topk    = 10                  # количество ближайших соседей

[defaults]
chunk_seconds = 6.0           # длительность аудио-чанка (сек)
batch_size    = 1             # размер батча для энкодера
filepath_col  = "filepath"    # колонка с путями в CSV
```

## Про providers

ONNX Runtime провайдеры определяются автоматически:
- CUDA доступна -> `["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]`
- CUDA недоступна -> `["CPUExecutionProvider"]`

Чтобы зафиксировать провайдеры явно, добавьте в `[encoder]`:
```toml
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

## Текущие конфиги

| Файл | Описание |
|------|----------|
| [`baseline.toml`](baseline.toml) | Baseline ONNX-энкодер + FAISS cosine kNN, topk=10 |

## Как добавить новый конфиг

1. Скопируйте `baseline.toml` -> `configs/inference/my_config.toml`
2. Измените нужные параметры
3. Запустите с флагом `--config`:
   ```bash
   speakerid-infer --csv data/test.csv --out sub.csv --config configs/inference/my_config.toml
   ```
