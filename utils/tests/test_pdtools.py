import itertools

import numpy as np
import pandas as pd
import pytest

import toolbox.utils.pdtools as pdt
from toolbox.utils.label import Keys
from toolbox.utils.pdtools import add_row
from toolbox.utils.wrap import name_tuple


def _best_image(a, b):
    """return 0 if a < b else 1 """
    return int(b.mean() > a.mean())


@pytest.fixture(scope='session')
def nan_table():
    """general DataTable for using in tests"""
    # FixMe: Why the name? Why such table, why not 'labeled_data_table'?
    import numpy as np
    shape = (10, 10)
    imLR = (np.ones(shape) * 1, np.ones(shape) * 2)
    best_view, best_data = [*zip('LR', imLR)][_best_image(*imLR)]

    import pandas as pd

    data1 = pd.DataFrame(map(Keys('data', 'view', 'kind', 'alg').label, [
        (imLR[0], 'L', 'image', 'cam'),
        (imLR[1], 'R', 'image', 'cam'),
        (best_data, best_view, 'image', 'GT'),
        (np.random.rand(*shape), best_view, 'image', 'rand')]))
    data2 = pd.DataFrame(map(Keys('data', 'view', 'frame', 'kind', 'alg').label, [
        (5, 'L', '01', 'image', 'cam'),
        (6, 'R', '01', 'image', 'cam'),
        (best_data, best_view, '01', 'image', 'GT'),
        (np.random.rand(*shape), best_view, '02', 'image', 'rand')]))

    db = pdt.DataTable(pd.concat([data1, data2]))
    db = db.set_index([*filter('data'.__ne__, db.columns)])

    return db


def test_group_iter(nan_table):
    dt = nan_table

    def compare_itr_df(group=None, data=None, index=None):
        grp_itr = pdt.group_iter(dt, group=group, data=data, index=index)
        if not index:
            index = list(set(dt.index.names).difference(group))
        if not data:
            data = list(dt.columns)
        # total_objects = sum([grp.shape[0] for _, grp in grp_itr])
        total_objects = []
        for _, grp in grp_itr:
            assert set(index) == set(grp.index.names), f"requested index {index} and given {grp.index.names} " \
                                                       "missmatch"
            try:
                grp_data = set(grp.columns)
            except AttributeError:
                assert len(data) == 1, "iterator returned a series but multiple data columns where given"
                grp_data = set([grp.name])
            assert grp_data == set(data), f"requested data {data} and given {grp_data} missmatch"
            total_objects.append(grp.shape[0])
        assert sum(total_objects) == dt.shape[0], "The original number of rows (data points) does" \
                                                  "not match the iterator output"

    compare_itr_df(group=['frame'])
    compare_itr_df(group=['alg'], data=['alg', 'data'])
    dt.reset_index('view')
    compare_itr_df(group=['view'])
    compare_itr_df(group=['view'], index=['alg'])


def test_index_like(multi_label_data_table):
    dt = multi_label_data_table
    # midx is a tuple
    midx = pdt.index_like(dt.index)
    # midx2 is a pandas MultiIndex
    midx2 = pdt.index_like(dt.index, as_tuple=False)
    assert midx == ('L', 'image', 'cam')
    assert list(midx2.levels) == ['L', 'image', 'cam'] and list(midx2.names) == ['view', 'kind', 'alg']


def test_iter_rows_dicts(multi_label_data_table):
    # please notice changed input which emphasizes method's functionality
    collection = [x for x in pdt.iter_rows_dicts(multi_label_data_table)]
    assert len(collection[0].keys()) == len(multi_label_data_table.columns)
    # assert each key from the method collection and col as the original column
    for key, col in zip(collection[0].keys(), multi_label_data_table.columns):
        assert key == col


# _TableMixIn tests
DS = pdt.DataSeries
DT = pdt.DataTable
PS = pd.Series
PD = pd.DataFrame


@pytest.fixture(autouse=True)
def table_unique_on_level():
    import numpy as np

    shape = (10, 10)
    imLR = (np.ones(shape) * 1, np.ones(shape) * 2)

    db = pdt.DataTable(map(Keys('data', 'view', 'kind', 'alg').label, [
        (imLR[0], 'L', 'image', 'cam1'),
        (imLR[0], 'S', 'image', 'cam2'),
        (imLR[1], 'R', 'image', 'cam3'),
    ]))

    db = db.set_index([*filter('data'.__ne__, db.columns)])
    return db


def test_squeeze_levels(multi_label_data_table: DataTable):
    cams = multi_label_data_table.cam  # image x [L x R]
    assert cams.squeeze_levels().index.nlevels == 1
    assert cams.squeeze_levels(keep='kind').index.nlevels == 2

    # drop rows with 'GT' and 'rand' to make ['alg', 'kind'] redundant levels
    dt = multi_label_data_table.drop(['GT', 'rand'], level='alg')
    assert dt.squeeze_levels().index.names == ['view']
    assert dt.squeeze_levels('alg').index.names == ['view', 'kind']
    assert dt.squeeze_levels(keep='alg').index.names == ['view', 'alg']


def test_find_level(multi_label_data_table):
    dt = multi_label_data_table
    assert dt.find_level('view') is True


def test_all_levels_names(multi_label_data_table):
    dt = multi_label_data_table
    dt.columns.set_names('data', inplace=True)
    assert dt.all_levels_names() == {'data', 'view', 'kind', 'alg'}


def test_named_levels(multi_label_data_table):
    dt = multi_label_data_table
    returned = dt.named_levels(levels=['view', 'alg'])
    returned_excluded = dt.named_levels(levels=['view', 'alg'], exclude=True)
    assert returned == ['view', 'alg']
    assert returned_excluded == ['kind']


def test_unstack_but(multi_label_data_table):
    dt = multi_label_data_table
    dt['data2'] = 3
    dt['data3'] = 2
    dt.columns = pd.MultiIndex.from_tuples([('root', col) for col in dt.columns], names=['lvl0', 'lvl1'])
    assert dt.xs('root', level='lvl0', axis=1).columns.values.tolist() == ['data', 'data2', 'data3']
    unstacked = dt.unstack_but(level=['view'])  # misspelled level does not raise error
    assert unstacked.index.names == ['view']
    assert set(unstacked.columns.names) == {'lvl0', 'lvl1', 'kind', 'alg'}


def test_stack_but(multi_label_data_table):
    dt = multi_label_data_table
    dt['data2'] = 3
    dt['data3'] = 2
    dt.columns = pd.MultiIndex.from_tuples([('root', col) for col in dt.columns], names=['lvl0', 'lvl1'])
    unstacked = dt.unstack_but(level=['view'])  # misspelled level does not raise error
    stacked = unstacked.stack_but(level=['lvl0'])
    assert stacked.columns.names == ['lvl0']


def test_rmi():
    custom_dt = DT(columns=['row1', 'row2', 'row3', 'col1', 'col2', 'col3', 'col4'])
    custom_dt.loc[len(custom_dt)] = np.nan
    custom_dt.loc[len(custom_dt)] = 'stay'
    custom_dt.loc[len(custom_dt)] = 'remove'
    custom_dt.set_index(['row1', 'row2', 'row3'], inplace=True)

    assert len(custom_dt.index) == 3
    assert len(custom_dt.rmi('remove').index) == 2
    assert len(custom_dt.rmi(row1='remove', row2='remove').index) == 2


def test_keep_level():
    custom_dt = DT(columns=['row1', 'row2', 'row3', 'col1', 'col2'],
                   data=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    custom_dt = custom_dt.set_index(['row1', 'row2', 'row3'])
    after_keep = custom_dt.keep_levels(level=['row1', 'row2'])
    assert len(after_keep.index) == 2


def sdt_general():
    """
    creates the following DataTable:
    columns:
        two levels :
            z with labels data1 and data2
            w with labels a b c d a b c d
            notice columns labels are not unique

    rows:
        two levels :
            x with labels 1 and 2
            y with labels 3 4 5 6 3 4 5 6
            notice rows labels are not unique

    ============================================

         z   data1          data2
    w       a  b  c  d     a  b  c  d
    x y
    1 3     1  2  3  4     5  6  7  8
      4     1  2  3  4     5  6  7  8
      5     1  2  3  4     5  6  7  8
      6     1  2  3  4     5  6  7  8
    2 3     1  2  3  4     5  6  7  8
      4     1  2  3  4     5  6  7  8
      5     1  2  3  4     5  6  7  8
      6     1  2  3  4     5  6  7  8

    ============================================

    Returns: newly created DataTable

    """

    dt = DT()
    dt['a'] = list(itertools.repeat(1, 8))
    dt['b'] = list(itertools.repeat(2, 8))
    dt['c'] = list(itertools.repeat(3, 8))
    dt['d'] = list(itertools.repeat(4, 8))
    dt['e'] = list(itertools.repeat(5, 8))
    dt['f'] = list(itertools.repeat(6, 8))
    dt['g'] = list(itertools.repeat(7, 8))
    dt['h'] = list(itertools.repeat(8, 8))

    dt.index = pd.MultiIndex.from_tuples(
        [(x, y) for x in ['1', '2']
         for y in ['3', '4', '5', '6']], names=['x', 'y'])

    dt.columns = pd.MultiIndex.from_tuples(
        [(x, y) for x in ['data1', 'data2']
         for y in ['a', 'b', 'c', 'd']], names=['z', 'w'])

    return dt


@pytest.fixture(name="sdt")
def sdt():
    return sdt_general()


def test_qix_basic(sdt):
    qixed = sdt.qix(w='a', y='4')
    assert not {'a'}.symmetric_difference(qixed.axes[1].get_level_values('w'))
    assert not {'4'}.symmetric_difference(qixed.axes[0].get_level_values('y'))
    qixed = sdt.qix(z='data1', keep='w', drop_level=True, axis=1)
    assert len(qixed.axes[1].names) == 1
    qixed = sdt.qix(z=['data1'], keep='w', drop_level=True, axis=1)
    assert len(qixed.axes[1].names) == 1
    qixed = sdt.qix('1', 'c', y=['3', '5'])
    assert len(qixed.index.get_level_values('x').difference(['1'])) == 0, "Only '1' left in x"
    assert len(qixed.columns.get_level_values('w').difference(['c'])) == 0, "Only 'c' left in w"
    assert len(qixed.index.get_level_values('y').difference(['3', '5'])) == 0, "Only '3', '5' left in w"
    # multiple values per level
    # not supposed to drop
    qixed = sdt.qix(w=['a', 'b'], y=['3', '5'], drop_level=True)
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 2
    # need to be dropped
    qixed = sdt.qix(w=['a'], y=['3'], drop_level=True)
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 1
    # multiplied values
    qixed = sdt.qix(w=['a', 'a'], y=['3', '3'], drop_level=True)
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 1
    # missing value in level
    qixed = sdt.qix(w=['a', 'g'], x=['1', '3'], drop_level=True)
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 1
    # empty DataTable
    qixed = sdt.qix(w=['g'], x=['3'])
    assert 'Empty' in qixed.__repr__()


def test_qix_specified(sdt):
    # list of drops
    qixed = sdt.qix(w=['a', 'a'], y=['3', '3'], drop_level=['w', 'y'])
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 1
    qixed = sdt.qix(w=['a', 'a'], y=['3', '3'], drop_level=['w'])
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 1
    # no drop
    qixed = sdt.qix(w=['a', 'b'], y=['3', '4'], drop_level=['w', 'y'])
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 2
    # keep levels
    # no drop
    qixed = sdt.qix(w=['a', 'a'], y=['3', '3'], drop_level=True, keep=['z', 'y', 'w', 'x'])
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 2
    # drop
    qixed = sdt.qix(w=['a', 'a'], y=['3', '3'], drop_level=True, keep=['z', 'w'])
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 2


def test_qix_anonymous(sdt):
    qixed = sdt.qix('a')
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 2
    qixed = sdt.qix('a', drop_level=True)
    assert len(qixed.axes[0].names) == 2
    assert len(qixed.axes[1].names) == 1
    # multiplied values with anonymous
    qixed = sdt.qix('a', w=['a', 'a'], y=['3', '3'], drop_level=True)
    assert len(qixed.axes[0].names) == 1
    assert len(qixed.axes[1].names) == 1
    # can't drop all levels of axis
    with pytest.raises(ValueError):
        sdt.qix('data1', 'a', drop_level=True)
    # missing level
    with pytest.raises(IndexError):
        sdt.qix('Im not there')


def test_freeze(multi_label_data_table):
    dt = multi_label_data_table
    initial_id = id(dt)
    dt.freeze()
    assert initial_id == hash(dt)


# pdtools methods tests
def test_kron():
    ds1 = DS(data=[1, 2, 3])
    ds2 = DS(data=[4, 5, 6])
    kroneckered = pdt.kron(ds1, ds2)
    assert kroneckered.loc[0, 1] == 5
    assert kroneckered.loc[1, 1] == 10


def test_outer():
    ds1 = DS(data=[1, 2, 3])
    ds2 = DS(data=[4, 5, 6])
    outered = pdt.outer(ds1, ds2)
    assert outered.loc[1, 1] == 10


def test_as_table():
    as_table = pdt.as_table
    ds = DS(dtype=float)
    dt = DT()
    ps = PS(dtype=float)
    pd = PD()
    assert id(as_table(ds)) == id(ds)
    assert id(as_table(dt)) == id(dt)
    assert type(as_table(ps)) is DS
    assert type(as_table(pd)) is DT
    assert type(as_table(ps)) is not PS
    assert type(as_table(pd)) is not PD
    try:
        as_table(2)
    except (TypeError, ValueError) as type_err:
        assert type_err


def test_append_col():
    append_col = pdt.append_col
    dt = DT(data=[[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], index=pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3)]))
    dt.columns = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3)])
    append_col(dt, col='4', values=[10, 11, 12])
    append_col(dt, col='4', values=[3, 1, 11])
    assert int(dt.loc[(1, 3), '4']) == 11


def test_sample(sdt):
    sample = pdt.sample
    assert len(sample(sdt, 2)) == 2
    assert len(sample(sdt, 2, shuffle=True)) == 2
    assert len(sample(sdt, 2, groups='y')) == 2 * 2
    assert len(sample(sdt, [1, 0, 3], groups='y')) == 2 * 3
    assert len(sample(sdt, 0.5)) == 4
    assert len(sample(sdt, 1.0)) == 8
    assert len(sample(sdt, 1)) == 1


def make_multi_index():
    """Create 3-levels multi-index"""
    return pd.MultiIndex.from_product([[1, 2], ['a', 'b'], ['one', 'two']],
                                      names=['num', 'let', 'word'])


@pytest.fixture
def mi():
    return make_multi_index()


def test_idx_mod_replace_at(mi: pd.MultiIndex):
    imod = pdt.IndexModifier(mi)
    mods = [(0, 10), (2, 'three')]  # define positional modifications
    unmod = set(range(len(mi[0])))  # unmodified indexes (to calc automatically)

    idx = mi[0]  # select multi index item
    res = imod.replace_at(idx, mods)

    for i, v in mods:
        unmod.remove(i)
        assert res[i] == v  # verify all the required modifications are done
    for i in unmod:
        assert res[i] == idx[i]  # varify the rest is unchanged


def test_idx_modify(mi: pd.MultiIndex):
    imod = pdt.IndexModifier(mi)
    idx = imod.name_index_tuple(mi[0])  # select test index item

    calc_key, calc_res = 'word', 'three'
    mods = {'num': 10, calc_key: lambda _: calc_res}  # define modifications

    unmod = set(mi.names).difference(mods)  # unmodified indexes

    # default mode: no named output and callables evaluated
    res = imod.modify(idx, mods)
    assert not hasattr(res, '_fields')  # default output is not named

    res = imod.modify(idx, mods, named=True)
    assert res._fields == tuple(mi.names)  # named output works
    assert getattr(res, calc_key) == calc_res  # callable has been evaluated
    for k, v in mods.items():
        if k is not calc_key:
            assert getattr(res, k) == mods[k]  # required fields are set
    for k in unmod:
        assert getattr(res, k) == getattr(idx, k)  # other fields remain intact

    # disable callables evaluation
    res = imod.modify(idx, mods, named=True, call=False)
    for k, v in mods.items():
        assert getattr(res, k) == mods[k]  # required fields are set (including callable!)
    for k in unmod:
        assert getattr(res, k) == getattr(idx, k)  # other fields remain intact


def test_indexers(mi: pd.MultiIndex):
    imod = pdt.IndexModifier(mi)
    mods = dict(A={'num': 10}, B={'let': 'X', 'word': 'three'})
    indexers = imod.group_indexers('Group', **mods, named=True)

    idx = imod.name_index_tuple(mi[0])
    grp = indexers(idx)
    grp._fields == tuple(mods)

    for fld, mod in mods.items():
        fld_idx = getattr(grp, fld)
        for lvl in mi.names:
            if lvl in mod:
                assert getattr(fld_idx, lvl) == mod[lvl]  # modified as requested
            else:
                assert idx[imod._pos[lvl]] == getattr(fld_idx, lvl)  # not modified

    indexer = imod.indexer(**mods['A'], named=True)
    assert indexer(idx) == imod.modify(idx, mods['A'], named=True)


def test_unbound_modifiers(show=False):
    imod = pdt.IndexModifier(["num", "let", "word"])
    idx = imod.name_index_tuple([1, 'a', 'one'])

    mods = dict(
        A={'num': lambda x: x.num + 2, 'let': 'Y'},
        B={'let': 'X', 'word': 'three'},
        C={'num': lambda x: x[0] + 2, 'let': 'Z'},
        D={'num': 10, 'let': 'W'}
    )

    for args, correct in [(
            dict(group_name='Group', call_named=True, named=False, A=mods['A'], B=mods['B']),
            name_tuple('Group', A=(3, 'Y', 'one'), B=(1, 'X', 'three'))
    ), (
            dict(group_name='Group', call_named=False, named=False, B=mods['B'], C=mods['C']),
            name_tuple('Group', B=(1, 'X', 'three'), C=(3, 'Z', 'one'))
    ), (
            dict(group_name='Group', call_named=True, named=True, B=mods['B']),
            name_tuple('IndexTuple', num=1, let='X', word='three')
    ), (
            dict(group_name='No', call_named=True, named=False, A=mods['A']),
            (3, 'Y', 'one')
    ), (
            dict(group_name='No', call_named=False, named=False, B=mods['B']),
            (1, 'X', 'three')
    ), (
            dict(group_name='', call_named=False, named=False, B=mods['B'], D=mods['D']),
            tuple([(1, 'X', 'three'), (10, 'W', 'one')])
    ), (
            dict(group_name='', call_named=False, named=True, B=mods['B'], D=mods['D']),
            tuple([(1, 'X', 'three'), (10, 'W', 'one')])
    ), (
            dict(group_name='Group', call_named=False, named=False, B=mods['B'], D=mods['D']),
            name_tuple('Group', B=(1, 'X', 'three'), D=(10, 'W', 'one'))
    ), (
            dict(group_name='Group', call_named=False, named=True, B=mods['B'], D=mods['D']),
            name_tuple('Group', B=(1, 'X', 'three'), D=(10, 'W', 'one'))
    )]:
        if show:
            from toolbox.utils.strings import dict_str
            print('-' * 40, dict_str(args, sep='\n', nested=False), '-' * 40, sep='\n')
        replace = pdt.IndexModifier.unbound_modifiers(**args)
        res = replace(imod, idx)

        if show:
            from timeit import repeat
            print(idx, ' --->\n', res, '\n')
            times = repeat("res = replace(imod, idx)", globals=locals(), number=(n := 1_000))
            print(f"Time: {min(map(lambda _: _ / n * 1e6, times)):.3f} µs")

        assert res == correct


# Datatable specific methods tests
def test_item_labels(multi_label_data_table):
    dt = multi_label_data_table
    idx_1 = dt.item_labels(1)  # second row
    assert idx_1.__repr__() == "<alg: 'cam', kind: 'image', view: 'R'>"


def test_add_items():
    def create_item(r, start, stop):
        return {f"k{c}": 10 * r + c for c in range(start, stop)}

    items = [create_item(r, 0 + r, 3 + r) for r in range(4)]
    table = pdt.DataTable([create_item(r, 1, 4) for r in range(100, 103)])

    for item in items:
        table.add_items(item)

def test_add_row(sdt):
    dt = sdt['data1'] # simpler datatable
    # data is dict - mapping between column and value
    new_dt = add_row(dt, {'a': 5,'b': 6,'c': 7,'d': 8}, index={'x':1, 'y': 3})
    assert len(new_dt) == len(dt) + 1
    # unnamed columns
    new_dt = add_row(dt, (5,6,7,8), index={'x': 1, 'y': 3})
    assert len(new_dt) == len(dt) + 1
    # data is scalar
    ds = add_row(dt['a'], 5, index={'x': 1, 'y': 3})
    assert len(ds) == len(dt['a']) + 1
    # data is a sequence
    ds = add_row(dt['a'], [5, 4, 5], index={'x': 1, 'y': 3})
    assert len(ds) == len(dt['a']) + 1
    assert isinstance(ds.iloc[-1], list)




if __name__ == '__main__':
    test_index_like()
    test_iter_rows_dicts()
    test_squeeze_levels()
    test_find_level()
    test_all_levels_names()
    test_named_levels()
    test_unstack_but()
    test_stack_but()
    test_rmi()
    test_keep_level()
    test_unbound_modifiers(show=True)
    test_qix_basic(sdt)
    test_qix_specified(sdt)
    test_qix_anonymous(sdt)
    test_freeze()
    test_kron()
    test_outer()
    test_as_table()
    test_append_col()
    test_item_labels()
