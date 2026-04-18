## Loaders

Лоадеры отвечают за превращение данных в numpy-волны фиксированного формата, которые дальше подаются в энкодер.

В текущем проекте есть 2 уровня:

1) **Waveform loader** - читает один аудиофайл и приводит его к единому формату.
2) **Dataset loader** - читает CSV-метаданные и итерируется по датасету, используя waveform loader.

### Конфигурация: два независимых измерения

Начиная с текущей версии, аудио-бэкенд и VAD настраиваются **отдельными секциями** в TOML:

```toml
[loader]
type      = "soundfile"   # или "torchaudio" — бэкенд чтения аудио
target_sr = 16000
clip      = false

[vad]
type             = "fireredvad"   # или "none" / отсутствие секции — VAD выключен
model_dir        = "models/FireRedVAD/VAD"
use_gpu          = true
speech_threshold = 0.4
min_speech_frame = 20
```

Это позволяет комбинировать любой лоадер с любым VAD:

| `[loader].type` | `[vad].type` | Результат |
|-----------------|--------------|-----------|
| `"soundfile"` | `"none"` / отсутствует | soundfile без VAD |
| `"soundfile"` | `"fireredvad"` | soundfile + FireRedVAD |
| `"torchaudio"` | `"none"` / отсутствует | torchaudio без VAD |
| `"torchaudio"` | `"fireredvad"` | torchaudio + FireRedVAD |

---

### TorchAudioWaveformLoader (torchaudio_loader.py)

Назначение: загрузить аудио-файл через `torchaudio` с бэкендом FFmpeg и привести его к формату, который ожидают следующие компоненты пайплайна.

Преимущества перед `SoundFileWaveformLoader`:

- Поддерживает значительно больше форматов (MP3, AAC, Opus, OGG, FLAC, WAV, M4A, …) через системный FFmpeg.
- Ресемплинг выполняется через `torchaudio.functional.resample` (нативный torch, без scipy).

Поведение:

- читает аудио через `torchaudio.load(..., backend="ffmpeg")`;
- downmix в mono (усреднение каналов через `tensor.mean(dim=0)`);
- ресемплит к `target_sr` через `torchaudio.functional.resample` если нужно;
- конвертирует в `numpy float32`;
- заменяет NaN/Inf на 0;
- опционально (флаг `clip`) клиппит амплитуды в [-1, 1] — **по умолчанию выключено**.

Требования:

- `torchaudio >= 2.0` (уже есть в зависимостях проекта).
- Системный бинарник `ffmpeg` должен быть доступен в `PATH` (установлен в Docker через пакет `ffmpeg`).

Параметры:

- `target_sr: int` — целевой sample rate (по умолчанию 16 000 Гц).
- `clip: bool = False` — включить клиппинг [-1, 1].

Использование в TOML-конфиге:

```toml
[loader]
type      = "torchaudio"
target_sr = 16000
clip      = false
```

---

### SoundFileWaveformLoader (soundfile.py)

Назначение: загрузить аудио-файл и привести его к формату, который ожидают следующие компоненты пайплайна.

Поведение:

- читает аудио через `soundfile`;
- downmix в mono (усреднение каналов);
- ресемплит к `target_sr` (по умолчанию 16k);
- заменяет NaN/Inf на 0;
- опционально (флаг `clip`) клиппит амплитуды в [-1, 1] - **по умолчанию выключено** (baseline-like).

Параметры:

- `target_sr: int` - целевой sample rate.
- `clip: bool = False` - включить клиппинг [-1, 1].

---

### VadWaveformLoader (vad_waveform_loader.py)

Назначение: обернуть любой waveform loader, вырезать только речевые сегменты через FireRedVAD и вернуть их конкатенацию.

Является drop-in заменой любого `WaveformLoader` — реализует тот же интерфейс.

Поведение:

- загружает аудио через `base_loader` (любой waveform loader — soundfile или torchaudio);
- прогоняет файл через `FireRedVad.detect()`;
- вырезает все речевые сегменты по временным меткам (`timestamps`) и конкатенирует их в один waveform;
- если VAD не нашёл речи — возвращает оригинальный waveform и логирует `WARNING`;
- накапливает суммарное время инференса VAD в атрибуте `vad_time_s` (читается runner'ом для `timing.json`).

Порядок обработки (важно):

```
файл на диске (любая длина)
    ↓  base_loader.load()              — читает ВЕСЬ файл (soundfile или torchaudio)
    ↓  FireRedVad.detect(path)         — VAD работает с ПОЛНЫМ файлом
    ↓  вырезаем речевые сегменты + конкатенируем
    ↓  VadWaveformLoader.load() возвращает speech-only waveform
    ↓  crop_or_pad_repeat_start(wav, chunk_seconds)  — обрезка до 6 сек
```

Ограничение `chunk_seconds = 6` применяется **после** VAD — в VAD идёт полный файл,
а 6 секунд берутся уже из чистой речи.

`VadWaveformLoader` создаётся автоматически реестром при `[vad] type = "fireredvad"`.
Напрямую инстанциировать его в конфиге не нужно.

Параметры секции `[vad]`:

- `type: str` - тип VAD. `"fireredvad"` — включить FireRedVAD; `"none"` или отсутствие секции — выключить.
- `model_dir: Path` - путь к директории с весами FireRedVAD (например `models/FireRedVAD/VAD`).
- `use_gpu: bool = false` - запускать VAD на GPU.
- `speech_threshold: float = 0.4` - порог вероятности для детекции речи (0 < x < 1).
  FireRedVAD выдаёт вероятность речи для каждого фрейма (~10 мс). Фреймы с вероятностью ≥ порога
  считаются речевыми. Ниже порог → больше захватывается (риск ложных срабатываний);
  выше → строже (риск пропустить тихую речь). Значение 0.4 — дефолт из документации FireRedVAD.
- `min_speech_frame: int = 20` - минимальное число последовательных речевых фреймов для
  признания сегмента речью. При фрейме ~10 мс это 200 мс. Короткие всплески короче этого порога игнорируются.

Использование в TOML-конфиге (soundfile + VAD):

```toml
[loader]
type      = "soundfile"
target_sr = 16000
clip      = false

[vad]
type             = "fireredvad"
model_dir        = "models/FireRedVAD/VAD"
use_gpu          = true
speech_threshold = 0.4
min_speech_frame = 20
```

Использование в TOML-конфиге (torchaudio + VAD):

```toml
[loader]
type      = "torchaudio"
target_sr = 16000
clip      = false

[vad]
type             = "fireredvad"
model_dir        = "models/FireRedVAD/VAD"
use_gpu          = true
speech_threshold = 0.4
min_speech_frame = 20
```

Скачать веса модели:

```bash
# Hugging Face
hf download FireRedTeam/FireRedVAD --local-dir models/FireRedVAD
```

Влияние на `timing.json`:

При использовании `[vad] type = "fireredvad"` в `timing.json` появляются дополнительные поля:

```json
{
  "inference_time_s": 58.023,
  "encoder_time_s":   45.678,
  "vad_time_s":       12.345,
  "search_time_s":    0.123,
  "total_time_s":     58.146
}
```

Важно: `inference_time_s` — это **суммарное** время фазы инференса, которое включает в себя
как работу VAD, так и работу энкодера:

```
inference_time_s = vad_time_s + encoder_time_s
```

`vad_time_s` и `encoder_time_s` — декомпозиция `inference_time_s`, а не отдельные фазы.
`search_time_s` (kNN-поиск) — отдельная фаза, не входит в `inference_time_s`.

При использовании лоадера без VAD поля `vad_time_s` в `timing.json` нет,
а `encoder_time_s ≈ inference_time_s`.

---

### CsvAudioDatasetLoader (csv_audio_dataset.py)

Назначение: прочитать таблицу файлов из CSV и детерминированно выдавать waveforms фиксированной длины.

Поведение:

- читает CSV (колонка с относительным путем к файлу + опционально `speaker_id`);
- резолвит пути относительно `root`;
- грузит waveform через waveform loader (любой из поддерживаемых, с VAD или без);
- нормализует длительность до ровно `chunk_seconds`:
  - если аудио длиннее - берётся начало (crop from start);
  - если короче - повторяется (repeat/tile);
  - это сделано для воспроизводимости оценки.

### Как добавить новый loader

Ориентируйтесь на протоколы в `src/dataton_tri_losya_49/pipeline/interfaces.py`.

- Для замены чтения/ресемплинга - добавляйте новый waveform loader (новый файл + ветка в `registry.py :: build_waveform_loader()`).
- Для нового VAD-бэкенда - добавляйте ветку `vad.type == "your_vad"` в `registry.py :: build_waveform_loader()`.
- Для изменения логики чтения метаданных/чанкинга - добавляйте новый dataset loader.
