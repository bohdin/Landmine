from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.router_detect import router as detect_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/test_images", StaticFiles(directory="data/test_images/images"), name="test_images")
templates = Jinja2Templates(directory="web/templates")

app.include_router(detect_router, prefix="/api")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/sim")
def sim(request: Request):
    return templates.TemplateResponse("sim.html", {"request": request})
