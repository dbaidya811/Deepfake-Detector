# Root app entrypoint for Hugging Face Spaces or simple uvicorn
# Exposes `app` imported from backend
from backend.app import app  # noqa: F401
