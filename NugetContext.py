from JSONClient import JSONClient

class NugetContext(object):
    def __init__(self):
        self.client = JSONClient()

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.client.__aexit__(type, value, traceback)
