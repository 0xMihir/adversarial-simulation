from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import annotations, cases, export
from .services import clf_service, texture_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    clf_service.init()
    texture_service.init()
    yield
    # atexit handles cache flush on shutdown


app = FastAPI(title="CISS Annotation Tool", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router, prefix="/api/cases", tags=["cases"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["annotations"])
app.include_router(export.router, prefix="/api/export", tags=["export"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
