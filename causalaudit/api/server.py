"""FastAPI application factory for CausalAudit.

The API is optional — the core CLI and library work without it.
Start with: ``causalaudit serve --port 8000``
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from causalaudit import __version__
from causalaudit.api.routes import router

logger = logging.getLogger("causalaudit.api")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance with routes and middleware attached.
    """
    from causalaudit.config import configure_logging, get_settings

    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="CausalAudit API",
        description=(
            "Causal drift detection for deployed ML models.  "
            "Detects structural changes in feature→label causal relationships."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for local dev; lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    async def _on_startup() -> None:
        logger.info(
            "CausalAudit API v%s starting on http://%s:%d",
            __version__, settings.api_host, settings.api_port,
        )

    logger.debug("FastAPI app created with %d routes", len(app.routes))
    return app
