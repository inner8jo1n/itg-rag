# Демо RAG

В [questions.json](questions.json) — 12 вопросов: 8 по базе и 4, на которые база
не должна отвечать. Документы описывают вымышленный магазин, чтобы сравнение
с моделью без контекста было проверяемым.

Для нового реального запуска:

```sh
make up
make ingest
make demo
make compare
```

Нужны ключи API в `.env`. Команды обращаются к настоящим embeddings, Weaviate
и LLM; стоимость определяется провайдером. Результаты попадут в `results/`.

Сохранённые артефакты можно читать без API:

- [answers.md](answers.md): все вопросы, ответы, источники и сравнение без RAG;
- [live_results.json](live_results.json): полный финальный live-прогон;
- [index_verification.json](index_verification.json): проверка повторной записи;
- [retrieval_experiments.json](retrieval_experiments.json): три профиля chunking;
- [before_tuning.json](before_tuning.json): первый прогон с порогом `0.45`;
- [experiments_before_tuning.json](experiments_before_tuning.json): исходные
  retrieval-эксперименты с тем же строгим порогом.
- [after_threshold_tuning.json](after_threshold_tuning.json): промежуточный прогон,
  который выявил перенос правила с настольных игр на другую категорию товара;
  после этого добавлена отдельная проверка достаточности контекста.

Записанные ответы — свидетельство конкретного запуска, а не гарантия качества
на других данных. `make verify-live` повторит полную проверку и обновит финальные
артефакты. Первый прогон до настройки сохранён отдельно.
