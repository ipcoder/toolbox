# Resource Management - Use Cases

First make sure you are familiar with the main [concepts](./concepts.md).

# Resources Life Timeline

### Resource Type Definition
Every resource is an instance of certain _resource type_ which must be defined first.

Resource types implemented as subclasses of pydantic model ``ResourseModel``
(thus may be called resource models).

```python
from inu.resman import ResourceModel, locatable
from inu.env import LOC_RES


@locatable(LOC_RES / 'houses')
class HouseModel(ResourceModel):
    rooms: int
    area: float
    ...

```
There are two stages in introducing resources into the framework:

1. Define a *locatable* Resource Model
2. Discover resources in the defined locations

### Creation
Resources can be created:
 1. explicitly as `*.yml` _files_
 2. explicitly as _instances_ of the corresponding `pydantic` model class
 3. implicitly during initialization of other resource as default attributes

### Discovery of Filed Resources
To be accessibly through the `resman` API the file-form resources must
be first _discovered_ and initialized as _instances_.

Discovery of resources is responsibility resource managers of their particular type. \
Those managers are created automatically _when a new class of resource Model is imported_.
``ModelsManager`` class can be used as a single access point to the resource management system.

For example, lets assume there is

```python
from inu.resman import ModelsManager  # initialize the resource management sub-system

# At this stage no specific resource types are registered in the ModelsManager!

assert not ModelsManager.list()
# Let's import model of datasets management resources defined by the ``datacast`` package.
from inu.datacast import models as dm

# Now all the resource Models (Datasource, Scheme, Dataset, DataCollection)
# from there are registered in the ModelManager
assert ModelsManager.list() > 3

# Manger of specific resource type can be found by its model class or name
ds_man = ModelsManager.get(dm.DatasetRM)
assert ds_man is ModelsManager.get('dataset')

# Discovery of its resources are can be requested
ds_man.discover()  # without arguments look in the default locations

# There is a shortcut version of this sequence:
ModelsManager.discover('dataset')

# Or, to discover resources from all the currently registered models
ModelsManager.discover()
```


## Scenarios

### Initialization

#### Access Resource Models

```python
from inu.datacast import ModelsManager, ResManager, find_resource

dsm: ResManager = ModelsManager.find("dataset")
assert dsm is None, "DatasetRM is not been imported yet"

from inu.datacast import models
# that imports among the other DatasetRM
dsm = find_resource('dataset')
assert isinstance(dsm, ResManager)
```

#### Scan Resource Folders

```python
from inu.datacast.models import DatasetRM
from inu.resman import ModelsManager

dsm = ModelsManager.find(DatasetRM)
assert dsm.list().empty, "Never scanned"
assert dsm.folders(), "Some Folders are defined with Model"
assert any(p.is_dir() for p in dsm.folders()), "Some folders must exist"

dsm.discover()  # assuming there are resources there
assert not dsm.list().empty



```


