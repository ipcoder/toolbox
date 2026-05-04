from __future__ import annotations
# FixMe: Do we still need that?

from typing import Optional

from iad.dataman.env import EnvLoc
from iad.dataman.resman import ModelsManager, ResourceModel  # initialize the resource management sub-system
from iad.core.logs import setup_logs, logger


mm = ModelsManager()


_log = logger('use_cases')
lof_func_enable = False


class MyRes(ResourceModel):
    description: str = "Hi"


class MyRes(ResourceModel):
    description: str = "Hi"


def log_func(f):
    def wrapper(*args, **kwargs):
        _log.warning(f"------------------- {f.__name__.upper()} ---------------------")
        return f(*args, **kwargs)
    return wrapper if lof_func_enable else f


def case_create_resource():
    class ResAModel(ResourceModel):
        description: Optional[str]

    ra1 = ResAModel('a1', description='Wrong')
    assert ra1.description


@log_func
def case_init_models_management():
    import iad.dataman.models as dm
    # At this stage no specific resource types are registered in the mm!
    mm._reset()      # but we reset anyway for robustness of the test
    assert not mm.list()
    # Let's import model of datasets management resources defined by the ``datasets`` bridge.
    mm.register_models(dm)
    # Now all the resource Models (Datasource, Scheme, Dataset, DataCollection)
    # from there are registered in the ModelManager
    assert len(mm.list()) > 3

    # Manger of specific resource type can be found by its model class or name
    datasets = mm.get(dm.DatasetRM)
    assert datasets is mm.get('dataset')  # search by name


@log_func
def case_resource_creation():

    from iad.dataman.models import DataSourceRM
    src = DataSourceRM.parse_file(EnvLoc.DATA / 'datasource.yml')  # config from yaml file - Standard

    data_root = EnvLoc.DATA.first_existing()
    # From config dict
    cfg = {'name': 'SourceName', 'root': data_root}
    src = DataSourceRM.parse_obj(cfg)  # 1. config from dict - pydantic
    src = DataSourceRM(**cfg)  # 2. config from arguments - Standard

    # Manual registration
    src = DataSourceRM('SourceName', root=data_root)  # name positional - Special init!
    assert str(src)
    assert not src.is_listed
    DataSourceRM._manager.add_resource(src)
    assert src.is_listed
    assert src == DataSourceRM('SourceName')  # Query config     # 4. config from query   - Special init!
    assert src == DataSourceRM.from_config('SourceName')  # ?? Alternative - Dedicated method for query!

    src = DataSourceRM(root=data_root)  # Optional name - default|special init (need?)
    assert repr(src)


@log_func
def case_resource_discoveries():
    # Resource Manager may be found also by a part of its name
    sources = mm.get('source')
    assert sources.discover()  # some has been registered

    # Manager contains its associated Resource Model as attribute
    res = sources.list('res')[0]
    assert sources.model(res.name) == res

    # There is a shortcut version of this sequence:
    assert mm.discover('source') == len(sources)

    # Or, to discover resources from all the currently registered models
    mm.discover()


@log_func
def scenario_work_with_collection():
    from iad.dataman import create_collection
    mm.discover()
    dc = create_collection('KITTI',
                           query=dict(subset='train'))

    mm.discover('scheme')
    src = mm.get('source')
    src.discover('/mnt/algo/DataSets/depth', only=False)
    mm.discover('scheme')

    dc = create_collection('MID14S', temp_cache=True)
    mm.discover('collect')


@log_func
def case_create_data_caster():
    from iad.dataman import create_caster
    my_dataset = create_caster(source=EnvLoc.DATA / 'my_dataset',
                               scheme='{width}_{height}_{color}_{material}.jpg',
                               filters=dict(color='BLUE',
                                            material=['steal', 'wood'],
                                            width=float(10).__le__,
                                            condition_square='height == width'))


@log_func
def scenario_typical_flow():
    from iad.dataman import create_caster, create_collection
    case_resource_discoveries()
    eth3d = create_caster('ETH3D')
    dc = create_collection(datasets=[eth3d])
    dc = create_collection(datasets=['MIDS12', 'FT3D'])
    dc = create_collection('SmallQualityEval')
