from pathlib import Path

from iad.dataman.models import DataSourceRM, DatasetRM
from iad.dataman.resman import ModelsManager, AutoScan

test_data = Path(__file__).parent / 'data'


def test_datasource(env_locs, tiny_stereo):
    config_path = test_data / 'datasource.yml'
    cm = ModelsManager.get(DataSourceRM)

    assert cm.find_resource('tiny', auto_scan=AutoScan.FIRST) == (ds := DataSourceRM('tiny'))
    assert not getattr(DataSourceRM.parse_file_to_dict(config_path), 'root', None) and getattr(ds, 'root')


def test_dataset(env_locs, tiny_stereo):
    ModelsManager.discover(DatasetRM)

    assert getattr(DatasetRM(
        name='test_ds',
        source='tiny',
        scheme='tiny'
    ).source, 'root')


def test_datacast():
    from iad.dataman import create_caster
    cst = create_caster('tiny', sample={'selection': 1})
    print(cst.collect())
