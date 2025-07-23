from __future__ import annotations

import contextlib
import logging
import time
from collections import namedtuple
from pathlib import Path
from typing import Callable, Union, Iterable, Literal, Sequence, Iterator


class Triggers:
    """Repository of triggers for actions based on certain conditions"""

    def __init__(self):
        self.triggers = []

    def add(self, action: Callable,
            cond: Union[Iterable, Callable, slice, int],
            *, use_context=False):
        """
        Register action to trigger and when the condition satisfied.
        :param action: Callable
        :param cond:
                    int - interval, triggers for >0 and never if interval is 0
                    slice - triggers when slice samples occur
                    Iterable - triggers if any value from the set defined by this iterable
                    Callable - any custom function: `f(i: int) -> bool`
        :param use_context: if True - expect to pass also a context dictionary into the action call
        """
        if isinstance(cond, slice):
            slc = cond
            step = 1 if slc.step is None else slc.step
            start = 0 if slc.start is None else slc.start
            stop = 2 ** 63 if slc.stop is None else slc.stop
            cond = lambda x: start <= x < stop and ((x - start) % step == 0)
        elif isinstance(cond, int):
            step = cond
            cond = lambda x: x > 0 and (x % step) == 0 if step else False
        elif hasattr(cond, '__iter__'):
            s = set(cond)
            cond = lambda x: x in s
        if not hasattr(cond, '__call__'):
            raise TypeError(f'Invalid condition type: {type(cond)}! use int, slice, Iterable ort Callable')

        self.triggers.append((cond, action, use_context))

    def invoke(self, status, context=None):
        """
        Invoke triggers conditioned by the given status
        :param status: usually time or simulation step
        :param context: any data from the caller context to be passed to signed actions, usually locals()
        """
        for condition, action, use_context in self.triggers:
            if condition(status):
                action(context) if use_context else action()


def progress(*args, **kwargs):
    """
    Progress bar iterator wrapper.
    Wraps any iterator adding progress bar functionality automatically detecting
    terminal and notebook environments.

    Example:
         for x in progress(range(10)):
            process(x)

    See tqdm.tqdm for all the supported arguments
    (if tqdm is not installed - just returns the iterator)

    """
    try:
        import tqdm
    except:
        return args[0]

    kwargs.setdefault('leave', False)
    run_env = detect_frontend()
    if run_env in ('terminal', 'ipython'):
        return tqdm.tqdm(*args, **kwargs)
    elif run_env == 'jupyter':
        return tqdm.tqdm_notebook(*args, **kwargs, )

    kwargs.setdefault('disable', True)
    return tqdm.tqdm(*args, **kwargs)


class Timer:
    """
    Measures and reports time passed in the context
    """

    # ToDo: @Ilya unite with `timed` as they share overlapping functionality!
    def __init__(self, msg='Completed in', out_func=print, *,
                 pre=None, sep='... ', active=True, min=0.0, fmt=' {:.3f}sec'):
        """
        Create timer object and pass (``msg`` + ``fmt``).format(time) into ``out_func`` when finished.

         - If ``msg`` contains own `{...}` formatting, ``fmt`` is not used.
         - Optionally ``pre`` message may be output in the beginning.

        :param msg: message to display when reporting with {} to format in time in seconds
        :param pre: preambule outputed before entering the body
        :param sep: separator between the preambule and the completing message
        :param out_func: function to be called for reporting
                         default - print, but may also use logging.info, warning, etc...
        :param min: minimal time to report
        :param active: if False deactivates - for silent modes
        """
        if isinstance(out_func, str):
            out_func, *log_level = out_func.rsplit('.', 1)
            log_level = (log_level[0] if log_level else 'debug').upper()
            log_level = logging._nameToLevel.copy()[log_level]
            logger = logging.getLogger(out_func)

            active = logger.isEnabledFor(log_level)
            out_func = lambda _: logger.log(log_level, _)
        elif active:  # if out_func is logger function deactivate if not enabled for this level
            out_obj = out_func.__self__  # object containing func (Logger for log.debug)
            if is_enabled := getattr(out_obj, 'isEnabledFor', None):
                active = is_enabled(out_obj.level)

        self.msg = '⏰' + msg + ('' if '{:' in msg else fmt)
        self.pre = pre
        self.out = out_func
        self.sep = sep
        self.active = active
        self.min = min

    def __enter__(self):
        if self.active:
            self.t0 = time.time()
            if self.pre:
                if self.out is print:
                    print(self.pre, end=self.sep)
                else:
                    self.out(self.pre + self.sep)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            dt = time.time() - self.t0
            if self.pre or dt >= self.min:
                self.out(self.msg.format(dt))


class TimePoints:
    """
    Time Points to place in the code and measure elapsed time between any two of them.
    Points can be named to be *reference* later to measure time elapsed from then.
    Points can be used for *measurements* by providing argument measure_from=<point_name>
    Timing can be enabled / disabled by setting the corresponding attribute
    """
    MT = namedtuple('MT', ['cpu', 'wall'])

    def __init__(self, enable=False, start=True, verb=True, progress=False,
                 clock: Literal['cpu', 'wall'] = 'wall'):
        """
        Initialize the time intervals measurements device.

        :param enable: initial state
        :param start: use this call as a 'start' point
        :param verb: verbosity
        :param progress: show the measurements along the progress
        :param clock: use 'wall' or 'cpu' clock when measuring `longest` and `total`
        """
        self.moments: dict[str, TimePoints.MT] = dict()
        self.enable = enable
        self.progress = progress
        self._verb = verb
        self._last = None
        self._clock = clock
        self._record = {}  # record of reports

        self._wt0 = time.time()
        if start:
            self.point('start')

    def verbose(self, verb=True):
        self._verb = verb

    def point(self, moment=None, *, measure_from=None,
              message='Elapsed {time: 7.3f}s from {from_moment} to {moment}'):
        """
         Add timing point into the code.

        :param moment: name of this point - if omitted would be point_N (N-number of points)
        :param measure_from: name of a point in the past to use. May be called 'last'
        :param message: Measurement result formatted string
        :return: None

        :Example:
            tm = TimePoints(True)
            ...
            tm.point()                        # 'point_0' point is created
            # some calculations
            ...
            tm.point(measure_from='last')     # no reference point is created here
            # some more calculations
            ...
            tm.point('big one', measure_from='last')  # 'big one' created, measured from 'point_0'
            # even more calculations
            ...
            tm.point()                        # 'point_1' created
            # insanely heavy calculation
            ...
            tm.point(measure_from='big one')  # measuring time elapsed from 'big_one'
            tm.point(measure_from='last')     # measuring time elapsed from 'point_1'

        """
        if not self.enable:
            return

        measure_from = measure_from or (
            'last' if self.moments and self.progress else None)

        if measure_from == 'last' and self.moments:
            measure_from = self._last

        if not moment and not measure_from:
            moment = 'point_' + str(len(self.moments))

        tm = self.MT(time.process_time(), time.time() - self._wt0)
        if moment:
            self.moments[moment] = tm
            self._last = moment

        if measure_from:
            prev_time = self.moments.get(measure_from, None)
            if prev_time:
                print(message.format(time=(tm.wall - prev_time.wall),
                                     moment=moment, from_moment=measure_from))

    def __call__(self, moment=None, *, measure_from=None,
                 message='Elapsed {time: 7.3f}s from {from_moment} to {moment}'):
        return self.enable and self.point(moment, measure_from=measure_from, message=message)

    def _measure_iter(self):
        return (getattr(x, self._clock) for x in self.moments.values())

    @property
    def longest(self):
        return max(self._measure_iter())

    @property
    def total(self):
        return sum(self._measure_iter())

    def report(self, title=None, *, show=True, record=False, min_time=0, **labels):
        """Create report on measured time points so far.

        :param title: Optional Title when printing
        :param show: if True - print it
            (also may pass columns to print from ['cpu', 'wall', 'dif'])
        :param record: if True add it to the record,
                       labels and current timestamp used as keys
        :param min_time: report only if total time exceeds this (sec)
        :param labels: additional labels associated with this report,
                    will be printed as header to the time stats,
                    and used as a key of the record
        :return: DataFrame with report
         """
        if not self.enable or self.total < min_time:
            return
        from pandas import DataFrame, option_context
        df = DataFrame({m: tm._asdict() for m, tm in self.moments.items()}).T
        df.index.name = 'TimePoints'
        df = df.diff()[1:]
        df['dif'] = df.cpu - df.wall
        df.loc['TOTAL'] = df.sum()

        if record:
            self._record = {{**labels, 'ts': time.time()}: df}

        if show:
            from toolbox.utils.strings import dict_str
            labels = labels or dict_str(labels, sep='|')
            title = title or ''
            (title or labels) and print(f"⏰ {title}: {labels}")
            if isinstance(show, str):
                show = [show]
            if isinstance(show, (list, tuple)):
                df = df[[*show]]
            with option_context('display.float_format', '{:,.3f}'.format):
                print(df)
        return df

    def summary(self, measure: Literal['cpu', 'wall'] = 'wall'):
        if not measure: return ''
        _msr = lambda _: getattr(_, measure)
        total = sum(map(_msr, self.moments.values()))
        longest = max(self.moments.items(), key=lambda _: _msr(_[1]))
        return f"[{measure}] total {total:.3f}s, longest: {longest[0]} {_msr(longest[1]):.3f}s"

    def records(self):
        # TODO: combine into one DataFrame with MiltiIndex
        raise NotImplementedError


@contextlib.contextmanager
def tqdm_joblib(**kwargs):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    import joblib
    from tqdm.auto import tqdm
    tqdm_object = tqdm(**kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def detect_frontend():
    """
    Return running environment:
    :return: 'jupyter notebook' | 'ipython terminal' | 'console'
    """

    from IPython import get_ipython

    ip = get_ipython()
    if ip:
        ip_type = str(type(ip))
        if 'ZMQInteractiveShell' in ip_type:
            return 'jupyter'
        if 'TerminalInteractiveShell' in ip_type:
            return 'ipython'
    return 'terminal'


def timed(report: Callable[[str], None] | logging.Logger | None = print, *,
          cond: Callable[[], bool] | str | int | bool = True, min=0,
          pre: Callable = None,
          form: Callable = '⏰{func_name}{args} call took {time:.3f} sec'.format
          ) -> Callable:
    """
    Create decorator to measure and report execution time of the function or method.

    If ``report`` is either a ``Logger`` instance or a Callable receiving a message report string.

    If Logger based reporter is enabled at the moment of the call, that also enable the time measurement.

    To have measurements logged AND a Callable condition, use specific logging function:
    ::
        @timed(report=logger.debug, cond=my_condition)
    ::

    Argument ``cond`` controls when the measuring is invoked:
        - ``False`` - skip creating decorator
        - ``True`` - always measure
        - ``Callable -> bool`` - function to decide each time (forbidden for Logger!)
        - a `str` - one of the logging levels names (only for Logger!)
        - an `int` - a logging level (only for Logger!)

    Argument ``fmt`` allows to control formatting of the timing reports

    Examples:
    ::
        @timed()                 # measure and print ALL the calls
        @timed(log, 'INFO')      # measure and log.info only if INFO level is enabled
        @timed(print, False)     # never
        @timed(log.debug, True)  # measure always, but report will appear only if debug is enabled

    Decorator returns wrapper function with attribute ``_original_func``.
    ::
        @timed()(func)._original_func is func

    To communicate timing results two argument are supported: ``report``, ``form``.
    They are separated is a matter of convenience to allow independent
    definition of "formatting" and "communication" methods:

        |  formatted = form(func_name=func_name, time=time, args=args, kwargs=kwargs)
        |  report(formatted)

    ``form`` function is supposed to pack reported fields the into form acceptable by ``report`` function.
    For example, format a string for logging:
    ::
        @timed(
            report=logging.getLogger('timings').debug,
            form="Function {func_name} executed in {time}s".format
        )

    However, ``report`` is optional, and ``form`` may implement all the required logic:
    ::
        events = {}

        def record_event(self, func_name, time, args, kwargs):
            events[func_name] = events.get(func_name, []).append(time)

        @timed(
            report=None,
            form=record_event, ...
        )


    :param report: Optional `Callable` with single argument produced by calling ``form``.
    :param cond: `False` - bypass wrapper, `True` - don't,
                 `Callable` - call before the function execution to decide if timing is needed.
    :param min: minimal time to report, in sec
    :param pre: optional str or format callable for message before the call
    :param form: a Callable receiving call description parameters for message formatting
    :return: decorator wrapping function if `cond` is not ``False`` or returning the original function.
    """
    from functools import wraps

    if isinstance(report, logging.Logger):
        logger = report
        # cond in this case is converted either to True or False
        if isinstance(cond, str):
            cond = logging.getLevelName(cond)
            if not isinstance(cond, int):
                raise ValueError(f'Unknown logging level name {cond}')

        if isinstance(cond, int):
            level, cond = cond, True
        elif cond is True:
            level = logging.CRITICAL
        elif cond is not False:
            raise ValueError(f"{cond=} for Logger report, expected <logging level>str|int|bool")

        report = lambda msg: logger.log(level, msg)

    if not isinstance(cond, (Callable, bool)):
        raise TypeError(f"Invalid {type(cond)=} when used with {report=}")

    def decorator(f):
        if cond is False:
            return f
        func_name = f.__qualname__

        @wraps(f)
        def wrapped(*args, **kwargs):
            if cond is True or cond():
                pre and report(pre(func_name=func_name, args=args, kwargs=kwargs)
                               if not isinstance(pre, str) else pre)
                t0 = time.time()
                res = f(*args, **kwargs)
                t = time.time() - t0
                if t > min:
                    formed = form(func_name=func_name, time=t, args=args, kwargs=kwargs)
                    report and report(formed)  # when report=None form() could be processor
                return res
            else:
                return f(*args, **kwargs)

        wrapped._original_func = f
        return wrapped

    return decorator


EXEC_OUT = Literal['list', 'generator', 'generator_unordered']


@contextlib.contextmanager
def exec_SPMD(func: Callable, total: int, *,
              jobs: dict | int | Literal[True, False, None],
              split_from=8, out: EXEC_OUT = 'list', show: str | dict = ''):
    """
    Single Program Multiple Data execution context.

    Given function and data size creates context to optionally apply it in parallel
    using ``joblib`` package, if size and jobs parameters meet certain criteria.

    Optionally shows progress bar, if desc or tqdm_par are provided
    Example:

    >>> with exec_SPMD(my_func, len(data), jobs=4) as (func, collect):
    ...     results = collect(func(item, **par) for item in data)

    Argument ``jobs`` recieves parallelization settings in different forms:
        - int > 1:  number of jops to run
        - True | None | -1: joblib automatically selects number of jobs
        - False | 0 | 1 - don't activate `joblib` machinery at all

    Argument ``data`` may be a size of data, or dataitem

    Argument ``out`` controls how the output is organized:
        "list" - collected into list (default)
        "generator" - generator yielding results as soon as they ready in the input order
        "generator_unordered" - same in arbitrary order (only for Parallel!)

    :param func: function to apply - will be returned as is, or as ``joblib.delayed(func)``
    :param jobs: parameters of ``joblib.Parallel`` | num of jobs | True|None|False
    :param total: number of data items to be processed
    :param split_from:
    :param out: produce the outputs as a list or generator
    :param show: Message to show with progress bar or dict with ``tqdm`` parameters
    :return: collecting object, wrapped func
    """
    if jobs is True or jobs is None:
        jobs = -1
    if not isinstance(jobs, dict):
        jobs_par = dict(n_jobs=jobs, return_as=out)
    else:
        jobs_par = jobs
        jobs = jobs_par.setdefault('n_jobs', -1)

    if not show:
        tqdm_par = dict(disable=True)
    elif isinstance(show, str):
        tqdm_par = dict(desc=show, total=total)
    else:
        tqdm_par = dict(total=total) | show

    if total >= split_from and (jobs < 0 or jobs > 1):
        from joblib import delayed, Parallel, cpu_count

        if jobs == -1:
            jobs = jobs_par['n_jobs'] = cpu_count(only_physical_cores=True)
        agg = Parallel(**jobs_par)
        func = delayed(func)

        tqdm = tqdm_joblib
        if desc := tqdm_par.get('desc', None):
            tqdm_par['desc'] = f"{desc} ({jobs=})"

    else:  # without joblib
        from tqdm import tqdm as tqdm

        def agg(results: Iterator):
            def updating_iter(itr):
                for x in itr:
                    pb.update(1)
                    yield x

            if show:
                results = updating_iter(results)
            if out == 'list':
                results = list(results)
            return results

    with tqdm(**tqdm_par) as pb:  # pb used in agg.updating_iter
        try:
            yield agg, func
        finally:
            pass


def scheme_from_labels(labels: dict):
    scheme = ""
    for key in labels.keys():
        scheme += f"{{{key}}}_"
    scheme += "{*}.npy"
    return scheme


class Dump(dict):
    """
    Class to dump data to a file, enables to trace location of the data along the flow of the algorithm.

    For example, if running an ANN model, the dump class can be used in different parts of the flow
    (encoder, decoder, etc.) to dump the data at different stages of the flow.
    The location of along the tree is passed during the initial call of the class in the module.

    The dump class can be used as a manager of dumping additional data from the algorithm,
    initialized with a config file that controls weather a certain dump call should be implemented or not.
    This way we can put the dump call in the code in various locations, and choose N
    out of the allocated calls from the config file.
    NOT IMPLEMENTED YET

    for example:
    >>> dump = Dump(config, root="path/to/root", labels={"label1": "value1", "label2": "value2"})
    >>> dump(data, labels={"label3": "value3"}) # data is dumped .../value1_value2_[label3=value3].npy
    >>> # or alternatively enables to update the labels
    >>> dump(data, labels={"label1":"value12", "label3":"value3"}) # .../value12_value2_[label3=value3].npy

    :param config: config file that controls the dump call, should have the following keys:
        active: bool, weather to activate the dump call or not,
        if False, the call will be ignored (default: False)
    :param root: root path to dump the data to,
        if None, the root path will be taken from the config file,
        if active is False, root is ignored (default: None),
        if active is True and root is None, an error is raised.
    :param labels: dict, labels to add to the path, not mendatory,
        but if None, labels should be passed in the call.
        labels can also be passed in the config file,
        and will be added (and overide) to the labels in the init.
        Labels in the call will overide default labels.
        scheme is build from the labels keys, the labels given in the init are the default labels.
    :param scheme: str, scheme to build the path from, if None, the scheme will be built from the labels keys.
    """

    def __init__(self, config, root=None, labels: dict = None, scheme=None):
        self.configured = None
        self.name_builder = None
        self.scheme = None
        self.config = config
        self.active = config.active if hasattr(config, 'active') else False
        self.labels = labels
        self.root = root
        self.scheme = scheme
        if self.active:
            self.configure()

    def configure(self, new_config=None):
        from toolbox.utils.paths import TransPath
        # TODO add
        #  1) support to selective config along the tree
        #  2) clearer way to configure
        if new_config is not None:
            self.config.update(new_config)
        self.root = Path(self.root) if self.root is not None else Path(self.config.root)
        self.labels = self.labels or {} | self.config.labels
        self.config.exist_ok = True if 'exist_ok' not in self.config.keys() else self.config.exist_ok
        self.scheme = self.scheme or scheme_from_labels(self.labels)
        self.name_builder = TransPath(self.scheme)
        self.configured = True

    def __call__(self, data=None, labels: dict = None, **kwargs):
        import os
        from toolbox.io import imsave

        if not self.active or data is None:
            return self.labels

        self.configured or self.configure()
        if labels is not None:
            self.labels = self.labels | labels
        else:
            assert self.labels, "labels should be passed in the call or in the init"
        # divide labels to default and anonymous
        defaults = {key: value for key, value in self.labels.items() if
                    key in self.name_builder.regex.categories}
        anonymous = {key: value for key, value in self.labels.items() if
                     key not in self.name_builder.regex.categories}
        name = self.name_builder(anonymous, **defaults)
        path = self.root / name
        if os.path.isfile(path) and not self.config.exist_ok:
            raise FileExistsError(f"File {path} already exists")
        path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        # TODO add support for other file types
        imsave(path, data)

    def fork(self, other: dict):
        return Dump(config=self.config, root=self.root, labels=other | self.labels, scheme=self.scheme)

    def __repr__(self):
        return f"Dump(config={self.config}, root={self.root}, labels={self.labels}, scheme={self.scheme})"
