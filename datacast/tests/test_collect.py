import logging

import numpy as np
import pytest

from toolbox.datacast.collect import DataCollection, SinkRepo
from toolbox.datacast.models import CollectionRM, SchemeRM

np.set_printoptions(precision=2, threshold=1000, suppress=True, linewidth=120)

logging.getLogger('datacast').setLevel(logging.DEBUG - 1)


def test_fetch(tiny_stereo):
    grp = tiny_stereo.qix(scene='Piano').rmi(alg='c2g')  # select rows scene='Piano' and gray images
    keep = ['view', 'kind', 'alg']  # index levels to keep
    disp_r, im_l, im_r = grp.select(
        [('R', 'disp'),  # fetch data for rows with specified values of
         ('L', 'image'),
         ('R', 'image')],
        levels=keep[:2],  # levels=['view', 'kind']
        keep_levels=keep  # keep ['view', 'kind', 'alg'] levels in the result
    ).fetch()

    assert disp_r.ndim == 2 and im_l.ndim == 3
    disp_l, disp_r_1 = grp.select(dict(view=['L', 'R'], kind='disp')).fetch()
    assert np.array_equal(disp_r, disp_r_1, equal_nan=True)

    disp_l_1, = grp.select(('L', 'disp'), ['view', 'kind']).fetch()
    assert isinstance(disp_l_1, np.ndarray)
    assert np.array_equal(disp_l, disp_l_1, equal_nan=True)

    # inplace
    orig_grp = grp.copy()
    fetch_no_inplace = grp.fetch(inplace=False)
    assert orig_grp.equals(grp) and not fetch_no_inplace.equals(grp)
    fetch_inplace = grp.fetch()  # inplace True is default
    assert not orig_grp.equals(grp) & fetch_inplace.equals(grp['data'])


@pytest.mark.skip(reason='imread jpg gray to rgb')
def test_iter(tiny_stereo):
    assert len(tiny_stereo) > 10

    g = next(tiny_stereo.iter('scene', trans=True)).grp
    assert len(g)
    assert len(g.GT) == 2
    assert len(g.RGB) == 2
    assert len(g.gray) == 2
    assert g.gray.L.item().shape == g.GT.R.item().shape


def test_mid_measure(tiny_stereo):
    from algutils.cache import Cacher, CacheMode
    cache_par = dict(name='cache_name',
                     folder=tiny_stereo.caster.root.parent.__str__())
    cacher = Cacher(**cache_par)

    disp_dc = tiny_stereo.filter(alg='GT', kind='disp').sample(slice(2))

    disp = disp_dc.measure_data(
        np.nanmax, out='max_disp', parallel='swift', cache=cacher
    )
    assert len(disp.max_disp) == len(disp_dc.db)
    assert cacher._file_name().exists()

    disp = disp_dc.measure_data(
        np.nanmax, out='max_disp', parallel='swift',
        cache=cache_par | {'mode': CacheMode.LOAD},
    )
    cacher._file_name().unlink()


def test_collection(tiny_stereo):
    from toolz import take

    def test_func(item):
        idx, df = item
        assert len(df) == 9
        assert ('L', 'disp') in df.index
        assert ('R', 'image') in df.index

    tests = tiny_stereo.filter(scene='Piano')
    assert 8 < len(tests) < len(tiny_stereo)
    list(map(test_func, take(2, tests.iter('scene', index=['view', 'kind']))))


def test_transforms(tiny_stereo):
    dc = tiny_stereo
    assert len(dc) > 10

    import pandas as pd
    pd.set_option('display.max_colwidth', 32)
    pd.set_option('display.width', 120)
    from toolbox.datacast.transtools import apply_column_transform
    from toolbox.datacast.transtools import Col
    itr = (dc.iter('scene', index=['kind', 'alg', 'view'], trans=False, out="frame")
           / (lambda g: g.xs('L', level='view', drop_level=False))
           / (lambda g: apply_column_transform(g, drop=False)))

    gid, grp = next(itr)
    assert not {Col.data, Col.read_trans, Col.path}.symmetric_difference(grp.columns)


def test_label_rules():
    from toolbox.datacast.labeled import LabelRules
    assert len(LabelRules.domains())
    rules = LabelRules('stereo')
    assert len(rules.labels)


def test_config(tiny_stereo):
    assert CollectionRM('tiny')
    assert CollectionRM(datasets='tiny')
    assert CollectionRM(datasets=['tiny'])


def test_mix(tiny_stereo):
    from copy import copy
    dup_tiny_stereo = copy(tiny_stereo)
    mixed_dc = DataCollection.mix([(tiny_stereo, 0.5)])
    assert len(mixed_dc) == 0.5 * len(tiny_stereo)
    mixed_dc = DataCollection.mix([
        (tiny_stereo, 0.5),
        (dup_tiny_stereo, 1.)
    ], shuffle=False)
    assert len(mixed_dc) == 1.5 * len(tiny_stereo)
    mixed_dc = DataCollection.mix([
        (tiny_stereo, 1.),
        (dup_tiny_stereo, 0)
    ])
    assert len(mixed_dc) == len(tiny_stereo)


def test_sink_repo():
    from inu.env import EnvLoc
    from algutils.paths import TransPath

    labels = {'dataset': 'FT3D',
              'scene': 'A_0000_0007',
              'gamma': '1',
              'avr': '1',
              'dpe_avr': '1',
              'alg': 'MoUNFcV4_3_3_base',
              'ver': '0.0',
              'cfg': 'c8a6',
              'ext': 'tif',
              'kind': 'feat',
              'sub': 'own',
              'view': 'L',
              'res': 2,
              'internal': 'encoder.left_feat_2',
              'data': 5.0}
    evals_sink = SinkRepo(dataset_or_scheme=SchemeRM('bench_evals'),
                          root=EnvLoc.EVALS.first_existing() or EnvLoc.EVALS.first)
    generated_path = evals_sink.path(no_tag=True, **labels)
    tp = TransPath(SchemeRM('bench_evals').search.pattern)
    parsed_labels = tp.regex.parse(str(generated_path), method='search')
    labels.pop('data', None)
    labels = {k: str(v) for k, v in labels.items()}
    assert parsed_labels == labels
