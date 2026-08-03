# abex — либа для анализа A/B тестов

Спека package structure. Код — потом.

## Структура пакета

```
abex/
├── pyproject.toml
├── README.md
├── src/abex/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py        # чтение из csv/parquet/db, приведение схемы
│   │   ├── validators.py     # dtypes, nulls, duplicates, cardinality checks
│   │   ├── profiling.py      # профиль метрики: dtype, распределение, skew,
│   │   │                     # доля нулей, outliers, sample size, число групп
│   │   │                     # (2 vs A/B/n), ratio vs non-ratio — вход для selector.
│   │   │                     # На больших данных считать через polars/duckdb lazy —
│   │   │                     # без полного материализации в pandas
│   │   └── outliers.py       # детект (IQR/MAD/z-score) + обработка:
│   │                         # winsorize/cap/trim/log-transform, до/после метрики
│   ├── design/
│   │   ├── power.py          # расчёт sample size, MDE, power
│   │   ├── srm.py            # sample ratio mismatch check
│   │   └── covariate_balance.py  # баланс ковариат между группами (pre-period)
│   ├── stats/                # каждая функция — чистая, + метаданные applicability
│   │   ├── frequentist.py    # t-test, z-test, chi2, mann-whitney;
│   │   │                     # A/B/n — ANOVA/Kruskal-Wallis + post-hoc с коррекцией
│   │   ├── bayesian.py       # bayesian A/B (beta-binomial, etc.)
│   │   ├── bootstrap.py      # bootstrap CI, permutation tests
│   │   ├── sequential.py     # sequential testing / always-valid p-values
│   │   ├── cuped.py          # variance reduction (CUPED)
│   │   ├── ratio.py          # ratio-метрики (revenue/user, CTR при повторных
│   │   │                     # показах на юзера) — delta method / linearization,
│   │   │                     # учёт кластеризации наблюдений внутри юзера
│   │   └── multiple_testing.py  # Bonferroni/BH коррекции при много метрик
│   ├── analysis/
│   │   ├── effect_size.py    # Cohen's d, relative lift, CI на эффект
│   │   ├── segments.py       # разбивка по срезам, поиск heterogeneous effects
│   │   ├── guardrails.py     # проверка guardrail-метрик (не только primary)
│   │   └── novelty.py        # детект novelty/primacy effect по времени
│   ├── selector/              # слой выбора метода — точка входа для агента
│   │   ├── registry.py        # реестр функций stats/design с метаданными
│   │   │                      # (тип метрики, распределение, min sample size,
│   │   │                      #  допущения, что возвращает)
│   │   ├── rules.py            # правила выбора: профиль данных → кандидаты методов
│   │   └── recommend.py        # publicAPI: recommend_test(profile) -> [(fn, why)]
│   ├── report.py               # сборка итогового отчёта: JSON-schema вывод
│   │                           # (metric, method, effect, ci, p_value, decision,
│   │                           #  agreement, warnings) + audit trail — почему
│   │                           # selector выбрал именно этот метод, для агента и людей
│   └── viz/
│       ├── distributions.py    # гистограммы/ECDF по группам
│       ├── timeseries.py       # динамика метрики и p-value по дням
│       └── outliers.py         # до/после обработки: box/scatter с подсветкой
│                               # отрезанных/winsorized точек, % удаленного per группа
├── tests/
│   ├── stats/                  # unit-тесты на известный effect size, по модулю
│   ├── design/                 # тесты на известный SRM / covariate imbalance
│   └── selector/                # тесты: профиль данных → ожидаемая рекомендация
└── examples/
    └── quickstart.ipynb
```

## Слой selector — вызов агентом

Агент не должен хардкодить "если метрика конверсия — юзай chi2". Вместо этого:

1. `data/profiling.py` считает профиль метрики/выборки: dtype (binary/count/continuous),
   размер групп, форма распределения (skew, наличие тяжёлых хвостов), доля выбросов,
   наличие pre-period данных (для CUPED/covariate balance), число сравниваемых метрик.
2. Каждая функция в `stats/*` и `design/*` регистрируется в `selector/registry.py` с
   декларативными метаданными: какие профили данных ей подходят, какие допущения
   нарушает при несоответствии, что возвращает (point estimate / CI / p-value / что-то ещё).
3. `selector/rules.py` — явные, читаемые правила сопоставления профиля с кандидатами
   (не ML, не черный ящик — детерминированная логика, которую можно аудировать).
4. `selector/recommend.py` — единая точка входа: агент передаёт профиль (или сырые
   данные, тогда сам вызовет profiling), получает ранжированный список подходящих
   функций с обоснованием и предупреждениями (напр. "выборка < MDE-порога",
   "сильный skew → предпочти bootstrap/mann-whitney над t-test").

Все функции в `stats/`, `design/`, `analysis/` остаются самостоятельно вызываемыми
напрямую (для человека/ноутбука) — `selector` не оборачивает их, а только рекомендует.
Реестр в `registry.py` — единственное место, которое надо обновлять при добавлении
нового метода, чтобы он сразу стал виден агенту.

### Политика при противоречащих результатах

Если top-2 рекомендованных метода дают разный вывод о значимости (напр. t-test
значим, bootstrap — нет), `recommend.py` это не скрывает и не усредняет:

- в ответе `recommend_test()` явное поле `agreement: bool` + список методов,
  которые разошлись, с их p-value/CI;
- при disagreement приоритет отдаётся методу с меньшим числом нарушенных допущений
  (по метаданным registry), не более "консервативному" p-value — правило фиксируется
  в `rules.py`, не в agent-side эвристике;
- report.py обязан пометить вывод как `"low_confidence"`, если disagreement был,
  и не подставлять единственное "да/нет значим" без этой пометки.

## Обязательные проверки перед тестом статзначимости

- SRM (sample ratio mismatch)
- Баланс ковариат между группами (pre-treatment periods)
- Достаточность sample size (power check)

## Обязательные проверки при анализе результата

- Множественные сравнения (если метрик > 1)
- Peeking / early stopping — используй sequential testing или фикс. длительность
- Novelty effect — смотри динамику эффекта во времени, не только agregate
- Guardrail-метрики — не только primary метрика роста

## Требования к коду

- Python 3.11+, type hints, dataclass для конфигов теста
- pandas + опция polars/duckdb для больших выборок
- Все stats-функции — чистые, без side effects, легко тестируемые на синтетике
- CV/тесты не нужны (не ML), но unit-тесты на известный effect size — обязательны
- Explicit random_state везде, где есть bootstrap/permutation
- Новый метод в stats/design без записи в selector/registry.py — неполный PR

## Тестирование и pre-commit

- Фреймворк — `pytest`. Раскладка тестов зеркалит `src/abex/` (см. `tests/`).
- Обязательный минимум на каждый stats/design модуль:
  - синтетика с known effect size → тест ловит эффект (не false negative)
  - синтетика с нулевым эффектом → тест не ловит ложный эффект на разумном alpha
  - граничные случаи: пустая/несбалансированная группа, NaN, одна группа
- `selector/`: тест на каждое правило rules.py — конкретный профиль данных →
  ожидаемый набор рекомендаций (не просто "не упало").
- `.pre-commit-config.yaml` в корне пакета, хук `pytest`:
  ```yaml
  repos:
    - repo: local
      hooks:
        - id: pytest
          name: pytest
          entry: pytest tests/ -q
          language: system
          pass_filenames: false
          always_run: true
  ```
- `pre-commit install` — обязательный шаг после клонирования (упомянуть в README).
  Коммит с падающими тестами не проходит; `--no-verify` не использовать.
- CI (если появится) дублирует тот же `pytest tests/` — pre-commit не единственная защита.

## Не включать

- AutoML/ML-модели для аплифта (uplift modeling) — отдельная либа, не MVP
- UI/dashboard — только Python API + опционально plotly-графики
