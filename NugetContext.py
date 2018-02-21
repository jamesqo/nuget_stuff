from JSONClient import JSONClient

class NugetContext(object):
    def __init__(self):
        self.client = JSONClient()

    async def __aenter__(self):
        self.client.__aenter__()

    async def __aexit__(self, type, value, traceback):
        self.client.__aexit__(type, value, traceback)
