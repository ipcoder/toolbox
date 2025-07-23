from pathlib import Path

from toolbox.datacast.models import DataSourceRM, DatasetRM
from toolbox.resman import ModelsManager
from toolbox.resman.resource import AutoScan

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
    ds = DatasetRM('tiny')

    from toolbox.datacast import DataCaster

    print(ds.dict())

    cst = DataCaster(**(ds.dict() | {'sample': 1}))
    print(cst.collect())
    ds.dict()
