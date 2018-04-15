import async_timeout
import asyncio
import json
import logging

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError
from json.decoder import JSONDecodeError

from utils.logging import StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

def is_404(exc):
    return isinstance(exc, ClientResponseError) and exc.code == 404

class JSONClient(object):
    def __init__(self):
        self._sess = None

    async def __aenter__(self):
        self._sess = await ClientSession().__aenter__()
        return self

    async def __aexit__(self, type_, value, traceback):
        await self._sess.__aexit__(type_, value, traceback)

    async def get(self, url, timeout=10):
        async with async_timeout.timeout(timeout):
            async with self._sess.get(url) as response:
                response.raise_for_status()
                text = await response.text()
                try:
                    return json.loads(text)
                except JSONDecodeError:
                    LOG.debug("Could not decode JSON from {}:\n{}", url, text)
                    raise

def _log_failure(url, excname, attemptno, delay):
    LOG.debug("GET {} failed with {}. Beginning attempt #{} in {}s...", url, excname, attemptno, delay)

class RetryClient(object):
    def __init__(self,
                 inner,
                 ok_filter,
                 retry_limit=5,
                 delay=1):
        self._inner = inner
        self._ok_filter = ok_filter
        self._retry_limit = retry_limit
        self._delay = delay

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, type_, value, traceback):
        await self._inner.__aexit__(type_, value, traceback)

    async def get(self, url, *args, **kwargs):
        for i in range(self._retry_limit):
            try:
                return await self._inner.get(url, *args, **kwargs)
            except Exception as exc: # pylint: disable=W0703
                if i == (self._retry_limit - 1) or not self._ok_filter(exc):
                    raise
                excname, attemptno, delay = type(exc).__name__, (i + 2), self._delay
                _log_failure(url, excname, attemptno, delay)
                await asyncio.sleep(delay)
