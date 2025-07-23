import re

import pytest
from pydantic import ValidationError

from toolbox.resman import ResNameError
from toolbox.resman.resource import ResModelError


def test_datasource(env_locs):
    # --------------- DataSourceRM ---------------
    existing_root = '/tmp'
    non_existing_root = '~/notexists'
    from toolbox.datacast.models import DataSourceRM
    # FROM root
    DataSourceRM(existing_root)
    assert DataSourceRM(non_existing_root)  # INVALID root -> PASS

    # Existing FT3D resource
    DataSourceRM('tiny')  # LOAD from existing
    DataSourceRM('tiny', root=existing_root)  # NEW root to existing 'FT3D' DataSourceRM (returns NEW RM)

    assert DataSourceRM('tiny', root=non_existing_root)  # INVALID root to existing 'FT3D' -> PASS

    # NOT Existing FT3D resource
    DataSourceRM._manager.remove('tiny')
    with pytest.raises(ValidationError):
        DataSourceRM('not_existing_tiny')  # NOT listed -> NO root -> FAIL

    assert DataSourceRM('tiny', root=non_existing_root)  # NOT listed -> INVALID root -> PASS
    DataSourceRM('tiny', root=existing_root)  # CREATE NEW DataSourceRM with VALID root


def test_scheme(env_locs):
    from toolbox.datacast.models import SchemeRM
    # --------------- SchemeRM ---------------
    SchemeRM(r'\w+.txt')  # valid pattern from SchemeRM constructor initialization
    with pytest.raises(re.error):
        SchemeRM(r'@#$*()')  # invalid pattern

    SchemeRM(search=r'\w+.txt')  # valid pattern from GuideScan constructor initialization
    SchemeRM(search=dict(pattern=r'\w+.txt'))  # valid pattern from GuideScan pattern field initialization
    SchemeRM('FT3D', search=r'\w+.txt')
    SchemeRM('FT3D', search=dict(pattern=r'\w+.txt'))

    with pytest.raises(ValidationError):  # INVALID search patterns
        SchemeRM(search=2)
    with pytest.raises(ResModelError):  # INVALID search patterns
        SchemeRM(search=dict(pattern='N0nv@l1dR3g3x'))
    with pytest.raises(ValidationError):  # INVALID search patterns
        SchemeRM('FT3D')  # no existing 'FT3D' -> no pattern -> FAIL
    with pytest.raises(ValidationError):  # INVALID search patterns
        SchemeRM('FT3D', search=2)
    with pytest.raises(ResModelError):  # INVALID search patterns
        SchemeRM('FT3D', search=dict(pattern='notapattern'))


def test_dataset(env_locs):
    """
    DatasetRM uses implicit logic on various scenarios.
    The SETUP phase creates [SchemeRM, SourceRM] X [layout, NO layout].
    Those are later used to test various use-cases.
    """
    from toolbox.datacast.models import DataSourceRM, SchemeRM, DatasetRM
    # --------------- DatasetRM ---------------
    # --------------- Register Resources ---------------
    scm_mgr = SchemeRM._manager
    scm_mgr.add_resource(SchemeRM('FT3D_WITHOUT_LAYOUT', search=dict(pattern=r'\w+.txt')))
    scm_mgr.add_resource(SchemeRM('FT3D_WITH_LAYOUT', search=dict(pattern=r'\w+.txt'), layout='FT3D'))

    dsrc_mgr = DataSourceRM._manager
    dsrc_mgr.add_resource(DataSourceRM('FT3D_WITHOUT_LAYOUT', root='/tmp'))
    dsrc_mgr.add_resource(DataSourceRM('FT3D_WITH_LAYOUT', root='/tmp', layout='FT3D'))

    # --------------- Use Cases ---------------

    DatasetRM('FT3D_WITH_LAYOUT')  # scheme, source from dataset name
    dsrc_mgr.remove(remove_all=True)
    with pytest.raises(ValidationError):  # schemes with no corresponding source
        DatasetRM(scheme='FT3D_WITHOUT_LAYOUT')
    with pytest.raises(ValidationError):
        DatasetRM(scheme='FT3D_WITH_LAYOUT')

    dsrc_mgr.add_resource(DataSourceRM('FT3D_WITH_LAYOUT', root='/tmp', layout='FT3D'))
    ds = DatasetRM(source='FT3D_WITH_LAYOUT')  # scheme, name from source name FT3D_WITH_LAYOUT
    assert ds.name == 'FT3D_WITH_LAYOUT'

    scm_mgr.remove(remove_all=True)
    scm_mgr.add_resource(SchemeRM('FT3D', search=dict(pattern=r'\w+.txt'), layout='FT3D'))
    ds2 = DatasetRM(name='FT3D', source='FT3D_WITH_LAYOUT')  # scheme from name
    assert ds2.scheme.name == 'FT3D'

    ds3 = DatasetRM(source='FT3D_WITH_LAYOUT')  # MASTER layout
    assert ds3.scheme.layout == 'FT3D' == ds3.scheme.name

    scm_mgr.remove(remove_all=True)
    scm_mgr.add_resource(SchemeRM('FT3D1', search=dict(pattern=r'\w+.txt'), layout='FT3D'))
    scm_mgr.add_resource(SchemeRM('FT3D2', search=dict(pattern=r'\w+.txt'), layout='FT3D'))
    with pytest.raises(ResModelError):
        DatasetRM(source='FT3D_WITH_LAYOUT')  # more than one discovered FT3D layouts -> NOT MASTER

    scm_mgr.remove(remove_all=True)
    with pytest.raises(ValidationError):
        DatasetRM(name='FAILED_FT3D', source='FT3D_WITHOUT_LAYOUT')  # scheme: name -> layout -> FAIL
    with pytest.raises(ValidationError):
        DatasetRM(source='FT3D_WITHOUT_LAYOUT')  # source.name -> name -> scheme with same name -> FAIL
    with pytest.raises(ResModelError):
        DatasetRM(source='FT3D_WITH_LAYOUT')  # source.name -> name -> scheme.name -> NO name==layout -> FAIL


def test_collection(env_locs):
    from toolbox.datacast.models import DataSourceRM, SchemeRM, CollectionRM, DatasetRM
    # --------------- CollectionRM ---------------
    DataSourceRM._manager.add_resource(DataSourceRM('FT3D', root='/tmp'))
    SchemeRM._manager.add_resource(SchemeRM('FT3D', search=dict(pattern=r'\w+')))
    col = CollectionRM(datasets=[{'source': 'FT3D', 'scheme': 'FT3D'}])  # create collection from dictionary
    assert not col.name  # cannot infer name

    with pytest.raises(ResNameError):
        CollectionRM._manager.add_resource(col)  # Cannot add resource without a name

    # Datasets as positional arguments
    ds1 = DatasetRM.from_config({'name': 'FT3D1', 'source': 'FT3D', 'scheme': 'FT3D'})
    ds2 = DatasetRM.from_config({'name': 'FT3D2', 'source': 'FT3D', 'scheme': 'FT3D'})
    DatasetRM._manager.add_resource(ds1)
    DatasetRM._manager.add_resource(ds2)
    assert [d.name for d in CollectionRM(ds1.name, ds2.name).datasets] == ['FT3D1', 'FT3D2']

    # create CollectionRM from various dataset forms
    col = CollectionRM(name='FT3D', datasets=[{'source': 'FT3D', 'scheme': 'FT3D'}, ds2, ds1.name])
    # explicit and implicit datasets initialization via CollectionRM constructor
    col2 = CollectionRM(datasets=[{'source': 'FT3D', 'scheme': 'FT3D'}, ds2, ds1.name])
    col3 = CollectionRM({'source': 'FT3D', 'scheme': 'FT3D'}, ds2, ds1.name)
    assert [d.name for d in col2.datasets] == [d.name for d in col3.datasets]

    CollectionRM._manager.remove(remove_all=True)
    DataSourceRM._manager.remove(remove_all=True)
    SchemeRM._manager.remove(remove_all=True)

    DataSourceRM._manager.add_resource(DataSourceRM('FT3D', root='/tmp'))
    SchemeRM._manager.add_resource(SchemeRM('FT3D', search=dict(pattern=r'\w+')))

    with pytest.raises(ResModelError):
        CollectionRM('FT3D', datasets=[])  # postponing datasets creation is NOT supported
    DataSourceRM._manager.add_resource(DataSourceRM('FT3D_OTHER', root='/tmp'))
    # dynamic Dataset creation
    CollectionRM('FT3D', datasets=[dict(name='FT3D', source='FT3D_OTHER', scheme='FT3D')])


if __name__ == '__main__':
    test_datasource()
    test_scheme()
    test_dataset()
    test_collection()
