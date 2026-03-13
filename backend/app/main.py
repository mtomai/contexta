import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pathlib import Path
from app.services.reranker import get_reranker
from app.config import get_settings
from app.routes import documents, chat, conversations, notebooks, agent_prompts, notes

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _quiet in ("httpcore", "httpx", "chromadb", "openai", "urllib3"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create necessary directories
Path(settings.uploads_path).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)

# ── Lifespan Manager ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    logger.info("Starting server. Pre-loading ReRanker model into RAM...")
    get_reranker()  # Loads the model into RAM at startup
    
    # (If you have other things to start, like database connection, put them here)
    
    yield  # <--- Here the server starts and waits for user requests
    
    # --- SHUTDOWN PHASE ---
    # This code will be executed when you press Ctrl+C or shutdown the server
    logger.info("Server shutdown in progress...")

# ── Initialize FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Contexta API",
    description="RAG-based document Q&A system with citations",
    version="1.0.0",
    lifespan=lifespan  # <--- Inserito qui durante l'unica inizializzazione
)

# ── No-Cache middleware for API responses ──────────────────────────────────────
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add no-cache middleware (must be added AFTER CORS to wrap correctly)
app.add_middleware(NoCacheMiddleware)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(notebooks.router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(agent_prompts.router, prefix="/api", tags=["agent-prompts"])
app.include_router(notes.router, prefix="/api", tags=["notes"])


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("API root called")
    return {"message": "Contexta API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}