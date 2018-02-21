import async_timeout
import json
import logging as log

from aiohttp import ClientSession
from json.decoder import JSONDecodeError

class JSONClient(object):
    def __init__(self):
        self._sess = ClientSession()

    async def __aenter__(self):
        await self._sess.__aenter__()
        return self

    async def __aexit__(self, type, value, traceback):
        await self._sess.__aexit__(type, value, traceback)

    async def get(self, url, timeout=10):
        async with async_timeout.timeout(timeout):
            async with self._sess.get(url) as response:
                response.raise_for_status()
                try:
                    return await response.json()
                except JSONDecodeError:
                    log.debug("Could not decode JSON from %s:\n%s", url, text)
                    raise
