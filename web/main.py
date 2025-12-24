import logging
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.database import init_db
from web.router_detect import router as detect_router
from web.router_missions import router as mission_router

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

app.mount("/static", StaticFiles(directory="web/static"), name="static")

test_images_dir = "data/test_images/images"
if os.path.isdir(test_images_dir):
    app.mount("/test_images", StaticFiles(directory=test_images_dir), name="test_images")
else:
    logger.warning("Skip mounting /test_images: directory not found (%s)", test_images_dir)
templates = Jinja2Templates(directory="web/templates")

app.include_router(detect_router, prefix="/api")
app.include_router(mission_router, prefix="/api")
init_db()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/sim")
def sim(request: Request):
    return templates.TemplateResponse("sim.html", {"request": request})
