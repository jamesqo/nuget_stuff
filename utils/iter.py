import sys

# Taken from https://stackoverflow.com/a/42379188/4077294
async def aenumerate(aiterable):
    i = 0
    async for x in aiterable:
        yield i, x
        i += 1

# Taken from https://stackoverflow.com/a/42379188/4077294
async def aislice(aiterable, *args):
    s = slice(*args)
    it = iter(range(s.start or 0, s.stop or sys.maxsize, s.step or 1))
    try:
        nexti = next(it)
    except StopIteration:
        return
    async for i, element in aenumerate(aiterable):
        if i == nexti:
            yield element
            try:
                nexti = next(it)
            except StopIteration:
                return
