# rate_limiter.py
import asyncio
import time
from collections import deque


class RateLimiter:
    def __init__(self, calls_per_minute: int, max_parallel: int = 25):
        self.calls_per_minute = calls_per_minute
        self.max_parallel = max_parallel
        self.window_size = 60  # 60 seconds = 1 minute
        self.timestamps = deque()
        self.lock = asyncio.Lock()
        # Semaphore for parallel request limiting
        self.parallel_semaphore = asyncio.Semaphore(max_parallel)

    async def _wait_for_slot(self):
        """
        Repeatedly check if there's space in the rate window.
        If at capacity, compute wait_time, release the lock, sleep,
        then re-check after waking up.
        """
        while True:
            async with self.lock:
                now = time.time()
                # Remove timestamps older than our window
                while (
                    self.timestamps and (now - self.timestamps[0]) >= self.window_size
                ):
                    self.timestamps.popleft()

                if len(self.timestamps) < self.calls_per_minute:
                    # There is room to make a call; record the timestamp and return
                    self.timestamps.append(now)
                    return
                else:
                    # We are at capacity, need to wait until the oldest one expires
                    wait_time = self.window_size - (now - self.timestamps[0])

            # Release the lock before sleeping so other coroutines
            # can also check if they have room
            await asyncio.sleep(wait_time)

    async def acquire(self):
        """
        1. Acquire the parallel semaphore (max_parallel).
        2. Wait until there's room in the calls_per_minute window.
        """
        await self.parallel_semaphore.acquire()
        try:
            await self._wait_for_slot()
        except:
            # If something goes wrong, release the parallel semaphore
            self.parallel_semaphore.release()
            raise

    async def release(self):
        """
        Release the parallel semaphore.
        """
        self.parallel_semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
