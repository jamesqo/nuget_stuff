import logging

from inspect import signature, stack
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
        super().__init__(logger, {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, log_kwargs = self.process(msg, kwargs)
            self.logger._log(level, BraceMessage(msg, args, kwargs), (), **log_kwargs) # pylint: disable=protected-access

    def process(self, msg, kwargs):
        param_names = signature(self.logger._log).parameters.keys() # pylint: disable=protected-access
        return msg, {name: kwargs[name] for name in param_names if name in kwargs}

LOG = StyleAdapter(logging.getLogger(__name__))

_funcs_logged = set()

def log_call(level=logging.DEBUG):
    funcname = stack()[1].function
    if funcname not in _funcs_logged:
        LOG.log(level, "{}() called", funcname)
        _funcs_logged.add(funcname)
