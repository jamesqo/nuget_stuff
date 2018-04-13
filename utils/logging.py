import logging

from inspect import getargspec, stack
from logging import LoggerAdapter

# Copied from https://stackoverflow.com/a/24683360/4077294
class BraceMessage(object):
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fmt).format(*self.args, **self.kwargs)

# Copied from https://stackoverflow.com/a/24683360/4077294
class StyleAdapter(LoggerAdapter):
    def __init__(self, logger):
        self.logger = logger

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, log_kwargs = self.process(msg, kwargs)
            self.logger._log(level, BraceMessage(msg, args, kwargs), (), 
                    **log_kwargs)

    def process(self, msg, kwargs):
        return msg, {key: kwargs[key] 
                for key in getargspec(self.logger._log).args[1:] if key in kwargs}

LOG = StyleAdapter(logging.getLogger(__name__))

def log_mcall(level=logging.DEBUG):
    method = stack()[1].function
    LOG.log(level, "{}() called", method)
