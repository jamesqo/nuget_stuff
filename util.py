import logging as log

from datetime import date, datetime, timedelta
from inspect import stack

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

def log_mcall(level=log.DEBUG):
    method = stack()[1].function
    log.log(level, "%s() called", method)

def tomorrow(as_datetime=False):
    result = date.today() + timedelta(days=1)
    if as_datetime:
        result = datetime.fromordinal(result.toordinal())
    return result
