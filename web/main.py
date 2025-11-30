from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from web.router_detect import router as detect_router

app = FastAPI()

# Підключаємо статику
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Підключаємо шаблони
templates = Jinja2Templates(directory="web/templates")

# Роут для API
app.include_router(detect_router, prefix="/api")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
