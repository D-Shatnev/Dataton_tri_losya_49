# Конфиги экспериментов (TOML)

В этой папке лежат **воспроизводимые** конфиги для dev-CLI:

```bash
python -m dataton_tri_losya_49 experiment --config configs/experiments/<name>.toml
```

Цель TOML-конфига - описать эксперимент так, чтобы можно было:

1) легко менять один компонент (например, энкодер) и сравнивать результаты,
2) не трогать код пайплайна,
3) получать артефакты (эмбеддинги/соседи/метрики) в предсказуемой папке.

## Важно про пути

Функция `load_experiment_config()` **не делает** пути абсолютными.
Это значит, что **все относительные пути в TOML интерпретируются относительно текущей рабочей директории**.
В типичном запуске это корень репозитория.

## Структура TOML

Конфиг состоит из пяти секций:

- `[experiment]` - имя эксперимента и папка для артефактов
- `[data]` - откуда брать аудио и как его резать/нормализовать
- `[encoder]` - чем получать эмбеддинги
- `[index]` - как искать ближайших соседей
- `[evaluation]` - какие метрики считать

Ниже - минимально рекомендуемый шаблон.

## Шаблон конфига

```toml
[experiment]
name = "my_experiment_name"           # строка, обязательное поле
out_dir = "experiments"               # куда писать артефакты (по умолчанию experiments)

[data]
csv = "data/test_public.csv"          # обязательное поле
root = "."                            # корень для относительных filepath из CSV
filepath_col = "filepath"             # имя колонки с путём к аудиофайлу
speaker_id_col = "speaker_id"         # (опционально) колонка с id спикера
chunk_seconds = 6.0                    # длительность чанка (сек)

[encoder]
type = "onnx"                         # сейчас поддержан только onnx
model_path = "models/baseline.onnx"   # обязательное поле
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
output_name = "embeddings"            # имя выхода ONNX-графа

[index]
backend = "faiss_ip"                  # сейчас поддержан faiss_ip
topk = 10                              # сколько соседей искать

[evaluation]
# если в CSV есть speaker_id_col -> он будет использован; иначе метрики будут пропущены
ks = [1, 5, 10]
# labels_npy = "data/labels.npy"       # (опционально) внешние метки, если нет speaker_id в CSV
```

## Что можно менять для экспериментов

### Быстро сравнить энкодеры (модель A/B)

Обычно меняется только секция `[encoder]`:

```toml
[encoder]
type = "onnx"
model_path = "models/encoder_A.onnx"
providers = ["CPUExecutionProvider"]
output_name = "embeddings"
```

Дальше - второй прогон с `models/encoder_B.onnx`.

### Поменять длительность аудио чанка

```toml
[data]
chunk_seconds = 3.0
```

Это влияет на то, какой фрагмент аудио будет подаваться на энкодер.

### Поменять top-k

```toml
[index]
topk = 50
```

Проверь, что в метриках `ks` не превышают `topk`.
