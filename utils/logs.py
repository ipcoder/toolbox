from __future__ import annotations

import logging
from logging import (getLogger, Formatter, FileHandler, StreamHandler, _nameToLevel as nameToLevel,
                     DEBUG, INFO, FATAL, ERROR, WARNING, WARN)
from pathlib import Path

# noinspection PyUnresolvedReferences
__all__ = ['getLogger', 'Formatter', 'FileHandler', 'StreamHandler', 'logging', 'logger',
           'nameToLevel', 'DEBUG', 'INFO', 'FATAL', 'ERROR', 'WARNING', 'WARN']

from typing import TypeVar

prf_fmt = '%(relativeCreated)5d|%(name)13s.%(funcName)-14s|%(levelname)7s|%(message)s'
reg_fmt = '▷%(relativeCreated)5d|%(name)-14s|%(levelname)7s🚦%(message)s'
def_fmt = reg_fmt


def set_format(profile=False):
    global def_fmt
    fmt = prf_fmt if profile else reg_fmt
    def_fmt = fmt
    log = getLogger('root')
    if not log.hasHandlers():
        handler = StreamHandler()
        log.addHandler(handler)
    log.handlers[0].setFormatter(Formatter(fmt))


def_datefmt = '%H:%M:%S'
T = TypeVar('T')


def _add_levels_attrs(obj: T) -> T:
    if not hasattr(obj, 'DEBUG'):  # already assigned
        for name, val in nameToLevel.items():
            setattr(obj, name, val)
    return obj


def module_log_file(file: str | Path):
    """Constructs default log file name from module file"""
    from toolbox.utils.filesproc import Locator
    out_folder = Locator('/tmp/ramdisk', '/tmp', envar='TEMP').first_existing()
    if not out_folder:
        from tempfile import mkdtemp
        out_folder = Path(mkdtemp())

    path = out_folder / 'logs' / f"{Path(file).name.split('.')[0]}.log"
    msg = f"Log file: {str(path)}"
    print(f"(!) ---> {msg}")
    logging.getLogger().info(f'Log file: {msg}')
    return path


def set_levels(logs_levels: dict[str, str | int] = None, *,
               debug=None, error=None, info=None, warn=None, **levels_logs: str):
    """
    Set levels for multiple logs:

    >>> set_levels(
    ...     {'deep_debug': 2},
    ...    debug = 'general',
    ...    info = ['scan', 'post'],
    ...    error ='root',
    ...    critical='another'
    ... )

    :param logs_levels:
    :param levels_logs: {level: log(s)}
    :param info:  log or logs name(s)
    :param debug: log or logs name(s)
    :param warn: log or logs name(s)
    :param error: log or logs name(s)
    """
    from toolbox.utils.short import drop_undef

    to_num = lambda lvl: nameToLevel[lvl.upper()] if isinstance(lvl, str) else lvl
    levels_logs |= drop_undef('debug', 'error', 'info', 'warn', ns=locals())
    levels_logs = {to_num(lvl): logs for lvl, logs in levels_logs.items()
                   } | {lvl: logs for logs, lvl in (logs_levels or {}).items()}

    from toolbox.utils import as_iter
    for level, logs in levels_logs.items():
        for log in as_iter(logs):
            getLogger(log).setLevel(level)


def logger(name: str = None, *, fmt=None, datefmt=None, level=None,
           add_handler: str | Path | bool | None = False):
    """
    Create stream logger with given settings, or return an existing one
    without altering.

    :param name: name to by accessed by
    :param fmt: handler formatting string
    :param level: logger level
    :param level: handler level
    :param add_handler: if True - adds default stream handler
    if str or Path - add
    :return: logger
    """
    log = getLogger(name)

    add_formatter = fmt or datefmt

    if isinstance(add_handler, (str, Path)):
        file = Path(add_handler).absolute()
        for h in log.handlers:
            if isinstance(h, FileHandler) and str(file) == h.baseFilename:
                break
        else:
            file.parent.mkdir(exist_ok=True)
            handler = FileHandler(file, mode='wt', encoding='utf-8')
            log.addHandler(handler)
            add_formatter = True
    elif add_handler is True or add_handler is None and not log.hasHandlers():
        handler = StreamHandler()
        log.addHandler(handler)
        add_formatter = True
    elif add_handler is not False:
        raise TypeError(f"Invalid {add_handler=}")

    if add_formatter:
        formatter = Formatter(fmt or def_fmt, datefmt=datefmt or def_datefmt)
        for handler in log.handlers:
            handler.setFormatter(formatter)

    if level:
        log.setLevel(level)

    return _add_levels_attrs(log)


def setup_logs(*, file=None, name_from=None, profile=False, **levels_logs):
    """
    High level function to quickly set basic logs config
    :param file: full path to the log file
    :param name_from: <name> from this path is used: ../logs/<name>.log
    :param levels_logs: {log_levels: loggers_names}
    :param profile: if True set profiling friendly format
    """
    # TODO: what if all three arguments not given ?
    if name_from:
        if file: raise ValueError("Use either `file` OR `name_from` argument!")
        file = module_log_file(name_from)

    if file:
        logger(None, add_handler=file)

    if levels_logs:
        set_levels(**levels_logs)

    set_format(profile)


def error(err: BaseException | type[BaseException], msg: str | None = None,
          fail: bool = True, level: int | str = 'ERROR',
          logger: logging.Logger | str | None = None):
    """
    Routes errors into logger and optionally raise Exception.

    :param err: Exception object ot type
    :param msg: message to log / throw
    :param fail: if True raise requested Exception
    :param level: logging level
    :param logger: logger or its name or None for root
    """
    if isinstance(err, type) and issubclass(err, BaseException):
        err = err(msg)  # create exception object given class, message
    elif msg:  # replace message in given exception object
        err = type(err)(msg)
    else:  # extract message from exception object
        msg = err.args

    if logger is None or isinstance(logger, str):
        logger = getLogger(logger)
    if not isinstance(level, int):
        level = nameToLevel[level]
    logger.log(level, msg)
    if fail:
        raise err
