import logging
import pytest

import pytest

from ..events import Timer, Triggers, timed


def test_timer():
    import logging
    log = logging.getLogger()
    with Timer("Testing Timer", log.debug):
        pass


def test_triggers():
    trs = Triggers()
    cases = {k: {'res': []} for k in ['interv', 'slice', 'set']}

    i = 0
    stop = 100

    def record_action(buf):
        def action():
            buf.append(i)

        return action

    step = 3
    cases['interv']['ref'] = [*range(step, stop, step)]
    trs.add(record_action(cases['interv']['res']), step)

    slc = slice(2, 55, 7)
    cases['slice']['ref'] = [*range(stop)][slc]
    trs.add(record_action(cases['slice']['res']), slc)

    cases['set']['ref'] = [8, 16, 77, 91]
    trs.add(record_action(cases['set']['res']), cases['set']['ref'])

    while i < stop:
        trs.invoke(i)
        i += 1

    for k, bs in cases.items():
        assert bs['ref'] == bs['res'], f"Trigger of type '{k}' has failed!"


@pytest.mark.skip(reason='Fails on timing')
def test_call_time():

    rep_str = []

    def report_to_str(msg):
        print(msg)
        rep_str.append(msg)

    def count_log_msg():
        return sum(map(bool, ss.getvalue().splitlines()))

    @timed(report_to_str, cond=True)
    def func():
        return

    from io import StringIO
    ss = StringIO()

    func()
    assert len(rep_str) == 1

    # -----------------------------------
    log = logging.getLogger('test')
    log.addHandler(logging.StreamHandler(ss))

    @timed(log, cond='INFO')
    def func():
        return

    func()
    assert count_log_msg() == 0

    log.setLevel(logging.DEBUG)
    func()
    assert count_log_msg() == 1

    # ------------------------------------
    events = []

    def record(func_name, time, **_):
        events.append({func_name: time})

    @timed(report=None, form=record)
    def func():
        return

    func()
    assert count_log_msg() == 1  # same as prev - not added!
    assert len(events) == 1
