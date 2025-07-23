from toolbox.utils.cache import *
from toolbox.utils.cache import _log


def test_modes(tmp_path):
    from toolbox.utils.cache import _log
    _log.setLevel(_log.DEBUG)

    def build(num=10, k=2, inc=1):
        return [
            CachedPipe.Source(range(num), f'source_{num}', copy=lambda x: x),
            CachedPipe.Map(lambda x: x * k, f'mul_{k}', copy=lambda x: x),
            CachedPipe.Map(lambda x: x + inc, f'add_{inc}', copy=lambda x: x),
        ]

    CachedPipe(build(5, 2, 1), folder=tmp_path, mode=CacheMode.CLEAR)

    res1 = [*CachedPipe(build(5, 2, 1), folder=tmp_path, mode=CacheMode.KEEP)]
    res2 = [*CachedPipe(build(5, 2, 1), folder=tmp_path, mode=CacheMode.LOAD)]
    assert res1 == res2

    _log.debug('--- Change last stage')
    res4 = [*CachedPipe(build(5, 2, 2), folder=tmp_path, mode=CacheMode.PASS)]
    assert res4 != res1


def test_filter(tmp_path):
    _log.setLevel(_log.DEBUG - 1)

    def build(num=10, k=2, inc=1, flt=4):
        return [
            CachedPipe.Source(range(num), f'source_{num}'),
            CachedPipe.Map(lambda x: x * k, f'mul_{k}'),
            CachedPipe.Filter(lambda x: x % flt == 0, f'flt_{flt}'),
            CachedPipe.Map(lambda x: x + inc, f'add_{inc}')
        ]

    res1 = [*CachedPipe(build(10, 2, 1), folder=tmp_path)]
    print(res1)
