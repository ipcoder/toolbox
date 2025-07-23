# Resources Management Framework

This framework provides basic tools to manage different kinds of generic _resources_.

Resource management is orthogonal to the meaning and usage of the resources.
It provides logistic services for their classifications, registrations, locations.

## Main Concepts and Components

Resource
: Entity defined as hierarchy of attributes, represented by (usually) YAML files,
  thus also referred as _configurations_.

Resource Model
: Semantic and syntactic structure of a particular _kind_ of
  resource is defined by the corresponding `pydantic` model.

YamlModel
: Subclass of `pydantic.Model` used as a base to implement `Resource Models`.

Configurations Manager
: Instances of a `ResManager` class managing configurations of one particular kind of
  resources identified by its `YamlModel`.

Models Manager
: A SW component managing the multiplicity of `Resource Models`.

### Compound Resources and Names

Being a hierarchy of attributes a resource may contain other resources:

- explicitly as sub-nodes
- implicitly by referencing

Resource referencing mechanism is based on their _names_,
which thus must be _unique_ in the context of particular resource kind,
(ensured by the corresponding `ResManager`).

Since the resource configuration semantic is controlled by its `pydantic` model,
providing a string for a node defined as a `ResourceModel` type,

## Class Diagram


```mermaid
classDiagram

    class ResourceModel{
       <<generic>>
       name
    }

    class ResManager {
        contains mapping from resource id of same type to resource configuration
        _resreg: dict[str, ResManager.RegInfo]
        find_resource(name)
    }

    class ModelsManager {
        <<singleton>>
        contains mapping of resources types to resource managers
        _res_managers: dict[Type[RT], ResManager[RT]]

    }

    ResManager o-- ResourceModel
    ModelsManager o-- ResManager


```

## Use Cases

### Creation of Resource

#### Create Unregistered Resource
```python
from inu.resman.resource import ResourceModel

class RT_X(ResourceModel):
  fld1: str
  fld2: str

RT_X('one', fld1='nice', fld2='very nice')
```

```mermaid
sequenceDiagram
    actor User
    User ->>+ ResourceModel[RT_X]: new('one', **cfg)
    ResourceModel[RT_X] ->>+ pydantic: init('one', *cfg)
    pydantic -->+ one: created one
    one -->>- ResourceModel[RT_X]: newly instantiated resource with name one
    ResourceModel[RT_X] -->>- User: one
```
---
#### Create Registered Resources

##### Without configuration
```python
from inu.resman.resource import ResourceModel

class RT_X(ResourceModel):
  fld1: str

RT_X('one')
```
```mermaid
sequenceDiagram
    actor User
    participant RT_X
    participant ResManager[RT_X]
    participant one[RT_X]

    User ->>+ RT_X: RT_X('one')
    RT_X ->>+ ResManager[RT_X]: find_resource('one')
    ResManager[RT_X] ->>+ one[RT_X]: one found on registry
    one[RT_X] -->>- ResManager[RT_X]: one returned
    ResManager[RT_X] -->>- RT_X: one returned
    RT_X ->> RT_X: Copy constructs one
    RT_X -->>- User: NEW copy of RT_X('one') returned to user

```
##### With configuration
```python
from inu.resman.resource import ResourceModel

class RT_X(ResourceModel):
  fld1: str

RT_X('one', fld2='nice')
```
```mermaid
sequenceDiagram
    actor User
    participant RT_X
    participant ResManager[RT_X]
    participant one[RT_X]

    User ->>+ RT_X: RT_X('one', fld2='nice')
    RT_X ->>+ ResManager[RT_X]: find_resource('one')
    ResManager[RT_X] ->>+ one[RT_X]: one found on registry
    one[RT_X] -->>- ResManager[RT_X]: one returned
    ResManager[RT_X] -->>- RT_X: one returned
    RT_X ->> RT_X: Finds that config is given, and warns the user that 'one' exists with other configuration
    RT_X ->> RT_X: Constructs one with NEW fld2 with value 'nice'
    RT_X -->>- User: RT_X('one') with altered fld2 returned to user

```

# FixMe: THIS diagram

[//]: # (```mermaid)

[//]: # (sequenceDiagram)

[//]: # (    actor User)

[//]: # (    participant ResMan)

[//]: # (    participant ResModel)

[//]: # (    participant Files)

[//]: # ()
[//]: # (    Note over User, File: Discover Resources Configurations)

[//]: # (        User ->>+ ResMan: discover&#40;ResModel&#41;)

[//]: # (            ResMan ->> +Files: search)

[//]: # (            Files ->>- ResMan: file_name)

[//]: # (    )
[//]: # (            ResMan ->> +ResModel: parse_file_to_dict&#40;file&#41;)

[//]: # (            ResModel ->> -ResMan: cfg)

[//]: # (    )
[//]: # (            ResMan ->> +ResModel: get_name&#40;cfg&#41;)

[//]: # (            ResModel ->> -ResMan: name = 'Two')

[//]: # (    )
[//]: # (            ResMan ->> ResMan: register&#40;name: &#40;file, cfg&#41;&#41;)

[//]: # (        ResMan ->> -User:)

[//]: # (    )
[//]: # (    Note over User, ResModel: Query Model - Unknown)

[//]: # (        User ->> +ResModel: ??? get&#40;name='One'&#41;)

[//]: # (            ResModel ->> +ResMan: find&#40;name='One'&#41;)

[//]: # (            ResMan ->> -ResModel: None)

[//]: # (        ResModel ->>-User: None)

[//]: # ()
[//]: # (```)

#### Query Registered Resource By Name

````python
r = ResModel('Two')
````

```mermaid
sequenceDiagram
        User ->> +ResModel: new(name='Two')
        ResModel ->> +ResMan: find(name='Two')
        ResMan ->> -ResModel: cfg

        create participant res_two
        ResModel ->>+ res_two: init('Two', *cfg)
            res_two ->> +ResModel: validate('Two', cfg)
            ResModel -->> -res_two: valid_cfg
        res_two -->>- ResModel: res_two
        ResModel -->> -User: res_two

```

#### Create Unregistered Resources Referencing Registered Resources

```python
from inu.resman import ResourceModel, ModelsManager, locatable

# Definition of the resources
@locatable('my/resources')
class ResModelB(ResourceModel):
    ...

class ResModel(ResourceModel):
    b: ResModelB
    ...

# Working with resources
ModelsManager.discover()   # discovers ResModelB('Two)
res_one = ResModel('one', b='Two', )
```

This scenario combines that of [Creation of Resource](#creation-of-resource)
and of [Query Registered Resource By Name](#query-registered-resource-by-name)

```mermaid
sequenceDiagram
    actor User
    participant ResModel

    Note over User, ResModel: Create resource 'One' REFERENCING <br> REGISTERED resource 'Two'

    User ->>+ ResModel: new('one', **cgf)
    create participant res_one
    ResModel ->>+ res_one: init('one', *cfg)
        res_one ->> +ResModel: validate(name='one', *cfg)
                ResModel ->> +ResModelB: new('Two')
                    ResModelB ->> +ResManB: find(name='Two')
                    ResManB ->> -ResModelB: cfg2

                    create participant resB_two
                    ResModelB ->>+ resB_two: init('Two', *cfg2)
                    resB_two ->> ResModelB: validate(name='Two', cfg2)
                    resB_two -->>- ResModelB: resB_two
                ResModelB -->> -ResModel: resB_two
        ResModel -->> -res_one: valid cfg
        res_one -->> -ResModel: res_one
    ResModel -->> -User: res_one

```

### Main Use Cases

```mermaid

sequenceDiagram
    actor User
    participant MM as Model<br>Manager

    create participant R1 as MyModel ::<br>ResourceModel
    User ->>+R1: import

    Note over User, MM: Register New Resource Model

    User ->> +User: @locatable(MyModel)
    R1 ->>+MM: register_model<br>(MyModel: ResourceModel)

    create participant CM as res_manager<br>[ MyModel ]
    MM ->>+ CM: new ResManager manager( MyModel )
    CM -->>- MM: res_manager [ MyModel ]
    MM -> MM: store in the _res_managers
    deactivate MM
    deactivate User

    Note over User,CM: Find ResManager class by resource model class or name
    User ->>+ MM: find<br>(model: class | name)
        MM -> MM: lookup in the _res_managers
    MM -->>- User: res_manager

    Note over User, CM: store and retrieve configurations
    User ->>+ CM: add_config(config, name, ...)
        CM -> CM: store in ConfigsMap
        deactivate CM

    User ->>+ CM: find_config(name)
        CM -> CM: lookup in ConfigsMap
        CM -->- User: config
```
