import asyncio
import time
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """
    Ограничивает количество одновременно обрабатываемых запросов.
    Остальные становятся в очередь.
    При превышении таймаута ожидания - возвращает 429.
    """
    def __init__(
        self,
        app,
        max_concurrent: int = 8,
        queue_timeout: float = 120.0,
        max_queue_size: int = 50,
        protected_paths: list = None # Если None - все пути защищены
    ):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_timeout = queue_timeout
        self.max_queue_size = max_queue_size
        self.protected_paths = protected_paths

        self._active = 0
        self._queued = 0
        self._total_processed = 0
        self._total_rejected = 0

    def _is_protected(self, path: str) -> bool:
        """
        Проверяет, нужно ли ограничивать этот endpoint
        """
        skip = ["/health", "/docs", "/openapi.json", "/redoc", "/metrics"]
        if any(path.startswith(s) for s in skip):
            return False
        
        if self.protected_paths is None:
            return True
        
        return any(path.startswith(p) for p in self.protected_paths)
    
    @property
    def stats(self) -> dict:
        return {
            "active": self._active,
            "queued": self._queued,
            "total_processed": self._total_processed,
            "total_rejected": self._total_rejected
        }
    
    async def dispatch(self, request: Request, call_next):
        if not self._is_protected(request.url.path):
            return await call_next(request)
        
        if self._queued >= self.max_queue_size:
            self._total_rejected += 1
            logger.warning(
                f"Очередь переполнена ({self._queued}/{self.max_queue_size})"
                f"Запрос отклонён: {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Сервер перегружен. Попробуйте позже.",
                    "queue_size": self._queued,
                    "active": self._active
                }
            )
        
        self._queued += 1
        queue_start = time.time()
        logger.info(
            f"Запрос в очереди: {request.url.path}"
            f"active={self._active}, queued={self._queued}"
        )

        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.queue_timeout
            )
        except asyncio.TimeoutError:
            self._queued -= 1
            self._total_rejected += 1
            wait_time = time.time() - queue_start
            logger.warning(
                f"Таймаут ожидания ({wait_time:.1f}s): {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": f"Время ожидания истекло ({self.queue_timeout}s). Попробуйте позже.",
                    "waited": round(wait_time, 1)
                }
            )
        
        self._queued -= 1
        self._active += 1
        wait_time = time.time() - queue_start

        if wait_time > 1.0:
            logger.info(f"Запрос ждал {wait_time:.1f}s в очереди: {request.url.path}")
        
        try:
            response = await call_next(request)
            return response
        finally:
            self._active -= 1
            self._total_processed += 1
            self.semaphore.release()


