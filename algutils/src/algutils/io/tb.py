"""
Read tensorboard logs
"""
from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.plugin_event_multiplexer import (
    event_accumulator, EventMultiplexer
)


def load_scalar(run: str | Path, tag, sub, timeit=True):
    """
    Return ALL the values of the scalar `tag/sub` from the `run`.
    May take seconds depending on the size!

    :param run: path to the run
    :param tag: tag under which scalar is know
    :param sub: specific sub-tag of the scalr
    :param timeit: if True measure and print time
    :return: DataFrame with (wall, time, step, `sub`) columns
    """
    from pandas import DataFrame, to_datetime
    from ..events import Timer

    mpx = EventMultiplexer(size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)
    run_tag_dir = str(Path(run) / f"{tag}_{sub}")
    mpx.AddRunsFromDirectory(run_tag_dir, name=sub)
    with Timer(f"Scalar {sub} loaded in {{:.3}}s", active=timeit):
        mpx.Reload()

        runs = mpx.Runs()
        assert len(runs) == 1, f"{runs=}"
        name = next(iter(runs))

        tensor_events = mpx.Tensors(name, tag)
        unpacked = ((event.wall_time, event.step, event.tensor_proto.float_val[0])
                    for event in tensor_events)
        df = DataFrame.from_records(unpacked, columns=['wall', 'step', sub])
        df['time'] = to_datetime(df.wall, unit='s', dayfirst=True)

    return df
