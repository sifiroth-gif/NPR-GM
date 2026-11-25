# Кандидаты, не результат испытаний (25.11.2026)

- Qwen/Qwen3.5-4B — Apache-2.0, post-trained; https://huggingface.co/Qwen/Qwen3.5-4B . Первый кандидат для мобильного теста. GGUF https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF .
- Qwen/Qwen3.5-9B — Apache-2.0, post-trained; https://huggingface.co/Qwen/Qwen3.5-9B . Компьютерная контрольная модель; качество ещё не измерено.
- RP fine-tune — отдельный кандидат после проверки лицензии, языка, совместимости tools/JSON и происхождения данных. Не выбирать по рейтингу чата; нужны реальные результаты на held-out.

Выбор пока условный: 4B vs 9B, GGUF Q4_K_M (если доступен и совместим), одинаковая версия llama.cpp, фиксированный SHA/квант, шаблон чата, seed, контекст 8192, max_tokens 1024 для пилота. Сравнить одинаковые prompt-only и инструменты. Не считать заявленное окно модели доступным на телефоне. Зафиксировать время до ответа, decode tok/s, RAM, нагрев, выгрузку ОС. Для обучения: Dettmers et al., QLoRA, https://arxiv.org/abs/2305.14314 ; метод уменьшает требования к памяти, но не гарантирует обучение на бесплатном GPU. Google Colab FAQ https://research.google.com/colaboratory/faq.html — бесплатный ресурс динамический, без гарантий. Не использовать как постоянный игровой сервер. Лицензии базовых моделей не заменяют проверку прав на данные.

## R2 — проверка рынка и предварительное решение (25.11.2026)
| Кандидат | Статус, лицензия | Роль |
|---|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Apache-2.0, post-trained, native 262144 tokens по карточке; [GGUF](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF), Q4_K_M | первичный прототип; около 2.8 GB файл по сторонним конверсиям, фактический размер фиксирует загрузчик |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache-2.0; [GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF), Q4_K_M около 1.4 GB по карточке | облегчённый резерв, не предполагается равное качество |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0; точная сборка GGUF и checksum пока не закреплены | контроль масштаба на ПК |
| [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | [GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF), семейство edge; E4B обозначает эффективный масштаб, а не размер файла | независимый контроль архитектуры; лицензия конкретного артефакта и ревизия при загрузке |
| [Qwen3-4B-RPG-Roleplay-V2](https://huggingface.co/Chun121/Qwen3-4B-RPG-Roleplay-V2) | MIT указана на карточке, англоязычная маркировка; происхождение обучающих данных и лицензии производных требуют отдельной проверки | RP-кандидат только для оценки после проверки происхождения; не базовая модель проекта |

[Официальный каталог Qwen](https://huggingface.co/Qwen/models) содержит Qwen3.8-27B, Flash-Next и более крупные модели; новое поколение не означает пригодность к смартфону. 4B выбран условно, не по результату RPG. Не использовать заявленное окно в 262K как бюджет устройства. Основной baseline 8192, CPU-smoke 4096/384 новых токенов, thinking=false для Qwen через chat_template_kwargs. Различие условий явно маркировать. Не сравнивать raw скорость разных машин. Реальный тест 24/24 и экспертная оценка ещё нужны.

[llama.cpp Android](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md): Termux/NDK; локальная работа, без API. [GitHub Actions](https://docs.github.com/en/actions/concepts/billing-and-usage): standard runners публичных repo бесплатны; только короткая задача публичного CI, не обучение или постоянный хостинг. [Colab FAQ](https://research.google.com/colaboratory/faq.html): бесплатные GPU и лимиты не гарантированы. R2 не закрывать до подтверждения реального целевого устройства и ревизий весов. Для CPU-smoke runtime b12678, официальный release asset с SHA256 проверяется; HF API фиксирует commit и LFS SHA256 при запуске, веса не публикуются. Ссылки проверены 25.11.2026.
## Первый реальный CPU-smoke — 25.11.2026
[GitHub Actions 45657238891 SUCCESS](https://github.com/sifiroth-gif/NPR-GM/actions/runs/45657238891), head 67c6b9259bb8d2c9acf2fe198314216a059e34bd. x86_64 runner, 4 CPU, RAM 16759435264 bytes. llama.cpp b12678 (c74c71a6f). Qwen3.5-4B-Q4_K_M, HF ggml-org/Qwen3.5-4B-GGUF revision 8d43d974efc48399cb58957ad49c8b26bf5a109c, SHA256 217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14, 3032091584 bytes (~3.03 GB). Проверка runtime и GGUF успешна. Контекст 4096, 4 threads, CPU, thinking=false, max_tokens=384. Dataset SHA256 9189ea23db2089ff043d3f2ccd3de74ade786d21b6217f771bcbc868d72ddfe8.
| Case | Full response seconds | Prompt tokens | Completion tokens | Auto errors |
|---|---:|---:|---:|---|
| E01 | 20.555 | 209 | 144 | 0 |
| E09 | 11.810 | 215 | 91 | 0 |
| E19 | 19.663 | 214 | 170 | 0 |
3/3 ответов, 3/3 автоматические проверки, медиана 19.663 s. Реальный inference, не fixture. Raw scenes намеренно не опубликованы; ручная семантическая/художественная оценка не выполнена. Замеры полного времени включают обработку входа и генерацию; не измерены отдельно TTFT/decode, peak memory, нагрев. Это доказательство работоспособности цепочки на серверном CPU, не доказательство качества модели, пригодности Pixel или успешного обучения. Полный R4 и физический R2 остаются открыты. Повторять загрузку без необходимости не требуется.

Целевое устройство утверждено: Pixel 10 Pro XL, 16 GB RAM, Tensor G5. Мобильная стратегия и пороги — HARDWARE.md. Первичный кандидат сохраняется; 9B не обязательна на телефоне.
