# Algorithmic Engines

## Introduction

Algorithm engine is a software component wrapping an arbitrary algorithm into
a standard interface including:
 1. Data interface (inputs and outputs)
 2. Configuration (algorithms parameters)
 3. State of the component as an execution unit


Abstract Base Class to subclass when defining Algorithms Engines.

# Basic Concepts

## Engine and Algorithm Separation
  - *Algorithm* represents a particular buseneess logic of the calculations
  - *Engine* is a vehicle to incorporate an algorithm into a data and excution flow.

### Terminology

Term *configuration* is used for ``Engine``, and parameters for ``Algorithm``.

### Subclassing ``AlgoEngine``

There are two levels of abstractions separating ``AlgoEngine`` from
a specific usefully engine:
  - Algorithm's *kind* abstraction
  - Algorithm's *implementation* abstraction

Specification of algorithm *kind* narrows to particular semantic
meaning of algorithms functionality, like disparity, segmentation, etc.
Algorithms of specific kind share same interface, in particular the
type of its ``Inputs`` and ``Outputs``.

Kind specification squires to:
    1. provide ``kind`` argument when subclassing ``AlgoEngine``
    2. define ``Inputs``and ``Outputs`` nested classes

Example:

```python
from inu.engines import AlgoEngine

class DispEngine(AlgoEngine, kind='disp'):
    class Inputs:
        imL: Image
        imR: Image

    class Outputs:
        disp: Image
        conf: Image = None   # optional
```

An actual algorithmic engine of disparity kind would inherit this abstract
interface class to implement those methods:
  - ``_process`` (REQUIRED) - algorithmic logic, returns Outputs)
  - ``default_config``      - class method, returns engine configuration)
  - ``_init_state``         - class method, returns engines state

```python
class DispEnginePDS(DispEngine):
     def full_config(self): ...
     def _create_engine(self, cfg): ...
     def _process(self, inputs: DispEngine.Inputs, par: dict): ...
```
Every fully defined Engine is automatically registered in ``EnginesRegistry``
where it can be found by its full name ("DispEnginePDS") or by its kind
and functional name ("PDS", kind="disp") using `EnginesRegistry.find_engine``.

Note, that subclassing ``AlgoEngine`` directly is not necessary,
intermediate inheritance is allowed as long as it does not specify *kind*.

As long as class remains abstract it may leave its *kind* undefined,
but any non-abstract Engine class must have its kind defined, either
through inheritance or by itself.

To summarise, there are two main restrictions:
 - Inheritance chain must not redefine kind.
 - Non-abstract Engines class must have kind defined
Exceptions are raised if those rules are violated.

### Batch Processing
This base class by default instantiates BatchScheduler to schedule
engine tasks execution, which requires a ``batch_processor``
function for its initialization. A simplistic implementation of
such function is provided in ``AlgoEngine._process_batch``.

One may either override this implementation or provide an instance
of different scheduler when creating an instance of Engine.
In the later case ``_process_batch`` is not used unless explicitly
done so, like in this example constructing completely
equivalent classes when classes``EngineA`` and ``EngineB``:

```python
class SomeEngine(AlgoEngine, kind='some'):
    class Inputs:
        x: int
        y: int
    # define abstract methods
    #   ...
```
scheduler = BatchScheduler(batch_size=16,
                           processor=AlgoEngine.process_batch)
EngineA = SomeEngine(scheduler=scheduler)
EngineB = SomeEngine(batch_size=16)


## Usage Scenarios
From user perspective Engine have two main stages: initialization and execution.

### Initialization
Engine instance it is constructed using
   - provided *engine configuration* (can not be later changed)
   - default *algorithm parameters* (can be updated per execution)

Implementing Engine requires ``default_config()`` class method which
return workable configuration, which may be updated by partial
configuration passed to the initialization.

### Execution

Applies the algorithm on specific input or batch of inputs.
Algorithm logic is defined by implementing ``_process()`` method.

Batch processing utilize default scheduler which can be replaced
for particular resources management strategy (GPU memory).

# Engines Registry

### Motivation
Access to `AlgoEngine` classes can be as usual achieved by importing the corresponding modules.
However, in case of multiplicity of such classes spead among different domains and modules,
the manual search for a required class can be quite cumbersome, especially in interactive sessions,
and also not suited for automation.

For this reason `inu.engines` package provides a simple *engines registry* API allowing querying
for available (registered) or required engines.

### Basic Usage

Engines registry is a singleton object named `engines` of `engines.Registry` class created in
`inu.engines.registry` and can be also imported from `inu.engines` namespace:

This registry object provides several methods to access the engines:
 - `engiens.list`: all the registered engine classes
 - `engines.find`: query by partial name and other attributes
 - `engines.klass`: get specific engine by partial name
 - `engines[num]`: get engine by its number in the *list*

   > ⚠ Read function's docs for the details about their particular arguments.

```python
from inu.engines import engines

df = engines.list()  # returns DataFrame with specified engines' attributes as columns
eng_cls = engines[2]  # return class of the second listed engine
engines.find(kind='prior_disp', fail=False)  # list specified engines
```
                  class_name        kind         module    package pfm
    0  EnhancePriorDispLTEn…  prior_disp      .prior_lt  disparity  LT
    1  SimpleNetC_PriorDisp…  prior_disp  .legacy.prior  disparity   T
    2  FillHoles_PriorDispT…  prior_disp  .legacy.prior  disparity   T
    3  Hourglass_PriorDispT…  prior_disp  .legacy.prior  disparity   T
    4  SimpleNet_PriorDispT…  prior_disp  .legacy.prior  disparity   T

```python
engines.find('Simple', kind='prior_disp')  # ... with `Simple` in the name
```
    class_name        kind         module    package pfm
    1  SimpleNetC_PriorDisp…  prior_disp  .legacy.prior  disparity   T
    4  SimpleNet_PriorDispT…  prior_disp  .legacy.prior  disparity   T

```python
NU_eng_cls = engines.klass('NU')  # returns `NU40DispEngine⋮disp` class
```

## Managing the Registry

Since engines may be defined in different modules and packages a special effort
is required to locate and manage a registry with available engined.

Automatic importing of all the engines classes in order to register them is very inefficient
and may unnecessary slow down imports even when engines are not actually used.

To solve this problem we instead introduce an off-line procedure for locating and catalogizing
available engines in special `YAML` catalog files.

Those catalogs are loaded when `engines` Registry is being initialized,
very fast and are prune of potential failures related to importing engine classes.

Of course, the drawback of this approach is that catalogs must be synchronized with
the code of the actually available engines when it is changed.

Package `inu.engines` contains module `create_catalogs.py` which implements
CLI functionality for creating those catalogs.

This utility can be executed manually or as part of git-commit hooks.

### Catalogs Implementation Details
Management of the engines catalogs are implemented using `inu.resman` package.
All the `engine.yml` file (under folders defined by the `EnvLoc.ENGINES` locator)
are `EnginesCatalogRM` resources.

`inu.engines.register.Registry` class is responsible to form the registry
from the found catalogs.

### Catalogs Maintenance
An **Engine Developer** is responsible to ensure every engine class is accessible
as `EnginesCatalogRM` resources:
 1. there is a `engines.yml` locatable through `EnvLoc.ENGINES` locator
 2. which contains engine's record modeled by `inu.engines.register.EngInfo`

Normally location of such engines catalogs reflects general architecture of a package,
and rarely requires changing `EnvLoc.ENGINES`.

Thus, routine development of engines happens under a given structure of the catalogs.
That allows to avoid manual maintenance of *version controlled* `engines.yml` files,
and relay instead on the `inu.engines.create_catalogs.py`.

   > ⚠ Note, that its current implementation can only **overwrite** or fail if such file exist.
   > Then all the manual changes are lost.

#### PIXI task
`algodev` project defines special `pixi` task for that:

```commandline
⚡ pixi run update_engines_catalogs
✨ Pixi task (update_engines_catalogs in default): python ./inu/engines/create_catalogs.py --overwrite --deep 1
Set environment from '/sp/code/algodev/.env, override=False'
Folder not found for Envar.TRAINED: Locator<EIA>$ALG_TRAINED=/home/ilyap/data/train []❌
Folder not found for Envar.BENCH: Locator<EIA>$ALG_BENCH=/home/ilyap/data/bench []❌
(!) ---> Log file: /tmp/logs/create_catalogs.log
Creating engine catalogs with options: overwrite=True, deep=1, dry=False
path=()
module=()
Found 1 in environment based paths
Creating for path /sp/code/algodev/inu/engines/disparity: overwrite=True, deep=1
Added 29 engines

```
