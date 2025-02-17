import keras
import logging
from functools import lru_cache


logger = logging.getLogger("bayesflow")


def _log(msg, *args, callback_fn: callable = print, **kwargs):
    if keras.backend.backend() == "jax":
        import jax

        def __log(*a, **k):
            callback_fn(msg.format(*a, **k))

        jax.debug.callback(__log, *args, **kwargs)
    else:
        callback_fn(msg.format(*args, **kwargs))


def critical(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.critical, **kwargs)


def debug(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.debug, **kwargs)


def error(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.error, **kwargs)


def exception(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.exception, **kwargs)


def info(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.info, **kwargs)


def log(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.log, **kwargs)


def warning(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logger.warning, **kwargs)


@lru_cache(None)
def warn_once(msg, *args, **kwargs):
    warning(msg, *args, **kwargs)
