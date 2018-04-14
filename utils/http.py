import async_timeout
import logging

from aiohttp import ClientSession
from json.decoder import JSONDecodeError

from utils.logging import StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

class JSONClient(object):
    async def __aenter__(self):
        self._sess = await ClientSession().__aenter__()
        return self

    async def __aexit__(self, type_, value, traceback):
        await self._sess.__aexit__(type_, value, traceback)

    async def get(self, url, timeout=10):
        async with async_timeout.timeout(timeout):
            async with self._sess.get(url) as response:
                response.raise_for_status()
                try:
                    return await response.json()
                except JSONDecodeError:
                    text = await response.text()
                    LOG.debug("Could not decode JSON from {}:\n{}", url, text)
                    raise
