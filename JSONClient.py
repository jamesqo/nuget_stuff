import json
import logging as log

from aiohttp import ClientSession
from async_timeout import timeout

class JSONClient(object):
    def __init__(self):
        self._sess = ClientSession()

    async def __aenter__(self):
        await self._sess.__aenter__()

    async def __aexit__(self, type, value, traceback):
        await self._sess.__aexit__(type, value, traceback)

    async def get(url, timeout=10):
        log.debug("GET %s", url)
        async with timeout(timeout):
            async with self._sess.get(url) as response:
                text = await response.text()
                try:
                    return json.loads(text)
                except JSONDecodeError:
                    log.debug("Could not decode as JSON:\n%s", text)
                    raise
