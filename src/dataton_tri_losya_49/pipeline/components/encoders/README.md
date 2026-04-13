## Encoders

Энкодер (encoder) — компонент пайплайна, который превращает аудиосигнал в векторные представления (эмбеддинги),
которые затем используются для поиска ближайших соседей (retrieval / speaker recognition).

В проекте энкодеры описываются контрактом (Protocol) `dataton_tri_losya_49.pipeline.interfaces.Encoder`.

### Контракт `Encoder`

Энкодер должен соблюдать следующие требования:

- **Вход**: батч waveforms `float32` формы `[B, T]`.
  - `B` — размер батча.
  - `T` — число сэмплов (длина сигнала). Внутри батча сигналы должны быть одной длины (обычно паддингом).
- **Частота дискретизации**: **16 kHz** (см. `SoundFileWaveformLoader.target_sr` и общий контракт пайплайна).
- **Выход**: эмбеддинги `float32` формы `[B, D]`.
  - `D` — размерность эмбеддинга (доступна через свойство `dim`).

Минимальный интерфейс:

```python
class Encoder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray: ...
```

### Текущая реализация: `OnnxEncoder`

`OnnxEncoder` выполняет инференс ONNX-модели через `onnxruntime`.

Параметры:

- `model_path`: путь к `.onnx` файлу.
- `providers`: список провайдеров ONNX Runtime (например `[
  "CUDAExecutionProvider", "CPUExecutionProvider"
]`).
- `output_name`: имя выхода, который содержит эмбеддинги (по умолчанию `"embeddings"`).

### Как добавить новый энкодер

Рекомендованный подход:

1) Добавить файл в этот пакет (например `hf_encoder.py`, `torch_encoder.py`, `onnx_quantized_encoder.py`).
2) Реализовать методы/свойства по контракту `Encoder`.
3) Экспортировать класс в `encoders/__init__.py`.
4) Подключить энкодер к экспериментам:
   - расширить фабрику `pipeline/runner.py::make_encoder()`;
   - расширить валидацию `pipeline/config.py::_validate_config()`;
   - добавить пример конфигурации в `configs/experiments/`.

Важно: пайплайн **не требует** наследования от `Encoder` (Protocol использует структурную типизацию).
