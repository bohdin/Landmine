# Landmine

Вебсервіс на FastAPI для детекції протипіхотних мін на зображеннях (завантаження користувачем або патчі у симуляторі).

## Структура
- `web/main.py` — FastAPI застосунок, роути `/` та `/sim`, статика/шаблони.
- `web/router_detect.py` — REST: `/api/detect`, `/api/history`, `/api/patches`.
- `web/templates/index.html` — завантаження зображень, вибір моделі, історія.
- `web/templates/sim.html` — симулятор маршруту, автодетекція патчів, стани clear/warn/danger.
- `web/static/uploaded`, `web/static/results` — збережені оригінали та результати.
- `core/` — інференс SSD, YOLO, WBF-ансамбль.
- `data/test_images/images` — приклади патчів для симулятора.

## Швидкий запуск локально
1) `pip install -r requirements.txt` (у віртуальному середовищі).
2) `uvicorn web.main:app --reload --host 0.0.0.0 --port 8000`.
3) Відкрити `http://localhost:8000` або `/sim`.

## API
- `POST /api/detect` — form-data: `file` (JPEG/PNG, ≤8 МБ), `mode` (`ssd` | `yolo` | `ensemble`), `threshold` (0–1), `draw_all` (показати бокси всіх моделей). Відповідь: `image_url`, `original_url`, `predictions`.
- `GET /api/history` — останні 20 викликів з посиланнями на результати.
- `GET /api/patches?count=N` — випадкові патчі з координатами для симулятора.

## Поведінка фронтенду
### `/`
- Форма для завантаження, вибір моделі/порогу, опція показати всі бокси.
- Блок історії для швидкого перегляду попередніх результатів.

### `/sim`
- Старт/Пауза, очистка маршруту, миттєвий «Додому».
- Вибір моделі/порогу для автодетекції патчів під час руху.
- Логіка загроз: score < (threshold+0.15) → `warn` (жовті мітки/треки), вище — `danger` (червоні).
- Патчі розкладаються кільцем навколо бази, щоб не потрапляти в початкову зону.

## Моделі
Необхідні файли: `models/ssd300.pth`, `models/yolo.onnx`. Базові пороги задаються у `web/models.py`; форму можна використовувати для зміни порогів у рантаймі.

## Збереження
`/api/detect` записує оригінал у `web/static/uploaded`, результат у `web/static/results`, додає запис у `web/history.json` (використовується на головній сторінці).

## Архітектурна схема (текстова)
```
Клієнт (браузер)
  ├─ / (завантаження) → POST /api/detect → core.Detector (SSD+YOLO+WBF) → static + history.json
  └─ /sim (маршрут) → GET /api/patches → рух дрона → POST /api/detect при наближенні → трек/мітки

Шари:
  - FastAPI: web/main.py, web/router_detect.py
  - Моделі: core/ssd_inference.py, core/yolo_inference.py, core/ensemble_inference.py
  - Шаблони/статичні файли: web/templates, web/static
  - Дані для симулятора: data/test_images/images
```
