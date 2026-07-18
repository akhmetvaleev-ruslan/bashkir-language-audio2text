# bash_language_spell

Проект по сбору и подготовке речевого корпуса башкирского языка (аудио + текст)
для обучения и оценки ASR-моделей (Whisper, Wav2Vec2/CTC). Данные собираются из
нескольких независимых источников — аудиокниги из Telegram-канала, стихи в
исполнении школьников (YouTube), диалектологические полевые записи в формате
ELAN, официальный датасет Mozilla Common Voice — и приводятся к единому
Common Voice-подобному формату (`clips/` + `train.tsv`), пригодному для
обучения через 🤗 `datasets`/`transformers`.

## Быстрый обзор папок с данными

| Папка | Что внутри | Откуда взялось |
|---|---|---|
| `books/` | 583 mp3, аудиокниги и рассказы, по одному файлу на главу/часть | Telegram-канал `bashaudiokitap`, `scrape_tg_books.py` |
| `pairs/` | `<Автор>/<Произведение>/audio/*.mp3` + `pdf/` с текстом | вручную отобранные пары «аудиокнига ↔ PDF текста» |
| `pairs_poems/` | `<Автор>/<Стихотворение>/audio/*.mp3` | `download_bashkir_audio.py` по ссылкам из `bashkir_poems_bashkir.csv` |
| `elang_data/` | `<локация>/<запись N>/*.eaf + *.wav` | полевые диалектологические записи, размечены в ELAN |
| `common_voice_pairs/` | `clips/*.wav` + `train.tsv` | сгенерировано `pairs_to_common_voice.py` из `pairs/` |
| `common_voice_pairs_poems/` | `clips/*.wav` + `train.tsv` | сгенерировано `pairs_to_common_voice.py` из `pairs_poems/` |
| `common_voice_elang/` | `clips/*.wav` + `train.tsv` | сгенерировано `eaf_to_common_voice.py` из `elang_data/` |
| `pdf/`, `kitaptar_collections/` | PDF/DjVu книг и собраний сочинений | `download_pdfs.py`, `download_kitaptar_collections.py` |
| `ctc-forced-aligner/` | сторонний инструмент (git-клон) для word-level форс-алайнмента mp3↔текст | используется при подготовке `pairs/`, `pairs_poems/` |
| `audio_bashkir/` | пусто | не используется ни одним скриптом; можно удалить |

Официальный `common_voice/` (Mozilla Common Voice, скачивается
`download_common_voice_mdc.py`/`download_dataset.py`) в этом чекауте
**отсутствует** — в корне лежит только осиротевший `train.tsv` (метаданные без
`clips/`), оставшийся от предыдущей попытки скачивания.

### Статистика по папкам с аудио (`stats.csv`)

В каждой папке, где реально лежат аудиофайлы, теперь есть `stats.csv` со
столбцами `author,book,records` — сводка «сколько записей на автора/книгу».
Файлы сгенерированы скриптом `generate_folder_stats.py` (см. ниже) и его можно
перезапустить в любой момент, если данные обновятся.

Смысл столбцов не везде буквален — структура папок неоднородна:

- **`books/`, `pairs/`, `pairs_poems/`** — `author`/`book` это реальные автор и
  произведение, `records` — число mp3-файлов (для `books/` метаданные автора
  берутся из `books_metadata.csv`; там, где Telegram-пост их не содержал,
  автор помечен как «Билгеһеҙ автор», а название — это исходное имя файла).
- **`common_voice_pairs/`, `common_voice_pairs_poems/`** — те же автор/книга,
  но `records` — число уже нарезанных по предложениям речевых клипов, а не
  число исходных mp3.
- **`common_voice_elang/`** — литературных авторов тут нет (это полевая
  речь), поэтому `author` = диктор, `book` = исходная запись-сессия
  (`source_file`), `records` — число клипов.
- **`elang_data/`** — `author` = диктор, извлечённый из тегов `[Имя]` в
  разметке `.eaf`; `book` = `<локация>/<запись N>`; `records` — число wav в
  этой сессии. Если в одной записи говорят несколько дикторов, сессия
  учитывается по разу на каждого диктора — поэтому сумма `records` по всей
  таблице больше, чем реальное число wav-файлов (185): это ожидаемо,
  колонка отвечает на вопрос «сколько записей связано с этим диктором», а не
  «сколько всего файлов».

## Скрипты в корне проекта

### Сбор исходных данных

- **`scrape_tg_books.py`** — тянет метаданные (и опционально сами mp3) из
  Telegram-канала `bashaudiokitap` через Telethon: разбирает подпись поста,
  имя файла и ID3/MP4-теги, складывает всё в `books_metadata.json/.csv`.
  Без `--download` работает быстро (только теги из первых 128 КБ файла), с
  `--download --download-dir books` — скачивает всё в `books/`.
- **`download_bashkir_audio.py`** — скачивает через `yt-dlp` аудио стихов из
  `bashkir_poems_bashkir.csv` (ссылки на YouTube) и раскладывает их в
  `pairs_poems/<Автор>/<Название>/audio/`. Идемпотентен: уже скачанные ролики
  отмечаются в `pairs_poems/.downloaded_bashkir_poems.txt`.
- **`download_common_voice_mdc.py`** — скачивает официальный Common Voice
  Bashkir 26.0 с Mozilla Data Collective (нужен `MDC_API_KEY`).
- **`download_dataset.py`** — альтернативный путь получения Common Voice —
  через Hugging Face `datasets` (`mozilla-foundation/common_voice_17_0`,
  языковой код `ba`); требует HF-токен и принятой лицензии на сайте датасета.
- **`download_pdfs.py`** — скачивает PDF книг, перечисленных в
  `author_works.csv`, с `kitaptar.bashkort.org` в папку `pdf/`.
- **`download_kitaptar_collections.py`** — через Playwright ищет на том же
  сайте собрания сочинений («Сочинения»/«Әҫәрҙәр»/«Һайланма») по всем
  авторам и скачивает найденные PDF в `kitaptar_collections/`.

### Поиск и сопоставление текстов

- **`find_author_works.py`** — по списку авторов (`authors.csv`) ищет их
  произведения в Algolia-поиске `kitaptar.bashkort.org`, результат —
  `author_works.csv` (ссылки на страницы книг, без самих файлов).
- **`find_wikisource_urls.py`** — сверяет стихи из `bashkir_poems_bashkir.csv`
  с полным списком заголовков категории «Шиғыр» на Wikisource и сохраняет
  только найденные совпадения-ссылки (текст не скачивает; найденное нужно
  проверять вручную на статус общественного достояния).
- **`search_kitaptar.py`** — ищет книги из `books_metadata.csv` (аудиокниги)
  на `kitaptar.bashkort.org`, чтобы понять, какие уже существуют там как
  текст; заголовок из имени файла предварительно чистится от пометок вида
  «... уҡый», «2 се бүлек» и т.п.

### Конвертация в Common Voice-формат

- **`pairs_to_common_voice.py`** — берёт результаты форс-алайнмента
  (`*.alignment.json` из `ctc-forced-aligner`) и парные mp3 из `pairs/` или
  `pairs_poems/`, режет их по предложениям на wav-клипы и пишет
  `common_voice_pairs(_poems)/clips/` + `train.tsv`.
- **`eaf_to_common_voice.py`** — то же самое, но из ELAN-разметки
  (`elang_data/*.eaf` + `*.wav`): нарезает по времени аннотированные реплики,
  распознаёт смену диктора по тегам `[Имя]` в тексте, пишет
  `common_voice_elang/clips/` + `train.tsv`.

### Обучение и оценка ASR

- **`whisper.py`** — дообучает Whisper (по умолчанию `openai/whisper-medium`)
  на одном или нескольких Common Voice-подобных каталогах сразу (объединяет
  их `train`/`eval`/`test` сплиты). Извлечение признаков кэшируется на диск
  по чанкам, чтобы падение на одном битом mp3 не убивало весь прогресс.
  По завершении обучения сам вызывает `evaluate_test_sets.py` на всех
  held-out test-сплитах.
- **`evaluate_test_sets.py`** — прогоняет уже готовый чекпоинт (локальный,
  из `whisper.py --output-dir`, или любой с HF Hub) по held-out `test.tsv`
  каждого каталога и считает WER/CER/MER/WIL. Автономный — можно
  перезапускать в любой момент без переобучения.
- **`transcribe_evaluate_whisper.py`** — то же, но для одного каталога и
  только для Whisper-чекпоинтов (модель обязана быть
  `WhisperForConditionalGeneration`, никакого автоопределения архитектуры).
- **`transcribe_evaluate_asr.py`** — универсальный вариант того же самого:
  сам определяет, CTC-модель перед ним (Wav2Vec2/Wav2Vec2-BERT/HuBERT) или
  seq2seq (Whisper), и транскрибирует соответствующим способом.

### Разное / утилиты

- **`list_paths.py`** — обходит указанную директорию и построчно пишет все
  пути файлов/папок в текстовый файл (использовался для `paths.txt`).
- **`generate_folder_stats.py`** — генерирует `stats.csv` (автор/книга/число
  записей) в каждой папке с аудио — см. раздел выше. Перезапускайте после
  изменений в `pairs/`, `pairs_poems/`, `books/`, `elang_data/` или
  пересборки `common_voice_*`.

## Типичный пайплайн от нуля до обученной модели

1. Собрать метаданные и аудио аудиокниг: `scrape_tg_books.py` → `books/`,
   `books_metadata.csv`.
2. (Опционально) найти и скачать тексты: `find_author_works.py` →
   `download_pdfs.py` → `pdf/`; либо `download_kitaptar_collections.py` →
   `kitaptar_collections/`.
3. Вручную сопоставить аудио с текстом произведения, разложить по
   `pairs/<Автор>/<Произведение>/{audio,pdf}`.
4. Прогнать `ctc-forced-aligner` по парам аудио+текст → получить
   `*.alignment.json` рядом с mp3.
5. `pairs_to_common_voice.py --input-dir pairs --output-dir common_voice_pairs`
   (аналогично для `pairs_poems`) → готовый `clips/` + `train.tsv`.
6. Для диалектологических записей: `eaf_to_common_voice.py --input-dir
   elang_data --output-dir common_voice_elang`.
7. (Опционально) скачать официальный Common Voice:
   `download_common_voice_mdc.py` или `download_dataset.py`.
8. Обучить: `whisper.py --data-dirs common_voice common_voice_elang
   common_voice_pairs common_voice_pairs_poems --output-dir whisper-ba`.
9. Оценить: `evaluate_test_sets.py --model-id whisper-ba --data-dirs ...`
   либо точечно `transcribe_evaluate_whisper.py` / `transcribe_evaluate_asr.py`.

## Прочие файлы в корне (не скрипты)

- `authors.csv`, `author_works.csv` — список авторов и найденных
  произведений на `kitaptar.bashkort.org`.
- `bashkir_poems_bashkir.csv`, `bashkir_poems_russian.csv` — списки стихов
  (исполнитель/автор/название/ссылка на YouTube) на башкирском и русском.
  **В git не публикуются** (см. `.gitignore`): среди исполнителей много детей
  (возрастные группы 0-5, 6-10, 6-11, 11-15 лет), а колонка «Исполнитель»
  содержит их настоящие имена — публикация такого списка на GitHub была бы
  публикацией персональных данных несовершеннолетних. Файлы остаются только
  локально и нужны для скриптов `download_bashkir_audio.py` /
  `find_wikisource_urls.py`. В репозиторий вместо них закоммичены
  `bashkir_poems_bashkir_public.csv` / `bashkir_poems_russian_public.csv` —
  те же данные без колонки «Исполнитель».
- `books_metadata.csv/.json` — результат `scrape_tg_books.py`.
- `wikisource_urls.csv`, `wikisource_matches.csv`,
  `wikisource_category_titles.json` — результаты сверки с Wikisource.
- `kitaptar_search_results.csv` — результат `search_kitaptar.py`.
- `pdf_audio_matches.md` — вручную аннотированный список совпадений
  PDF ↔ аудиокнига.
- `merge_log*.txt`, `merge_state.txt`, `inspect.txt`, `poems_list.txt`,
  `convert_pairs*.log` — рабочие логи ручной чистки дублей авторов/названий
  папок и логи прогонов `pairs_to_common_voice.py`; исторические, для
  повторного запуска скриптов не нужны.
- `check.ipynb` — черновой notebook для просмотра `kitaptar_search_results.csv`.
- `train.tsv` (в корне) — осиротевшие метаданные official Common Voice без
  соответствующей папки `clips/` (см. предупреждение выше).
- `tg_session.session` — сессия Telethon для `scrape_tg_books.py` (содержит
  авторизацию Telegram-аккаунта — не публикуйте и не коммитьте этот файл).
