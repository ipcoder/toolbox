from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Literal, get_args, Collection, Iterable

from toolbox.engines.core import AlgoEngine
from inu.env import EnvLoc
from toolbox.param.models import YamlModel, pyd
from toolbox.resman.resource import ResourceModel, locatable
from toolbox.utils import drop_undef
from toolbox.utils.filesproc import PyModuleLocator
from toolbox.utils.logs import getLogger, error

_log = getLogger('engines')
ENGINES_YML = 'engines.yml'  # fixed name for any engines catalog


# noinspection PyEnum
class EngInfo(YamlModel, extra=pyd.Extra.forbid):
    """
    Pydantic Model with information about a single Engine.

    Validates engine name consistency and / or completes some fields from the name.

    Can return config as dict in the form suitable for saving in engines.yml
    """
    class_name: str
    kind: str
    module: str
    name: str | None = None  # part of the class_name
    package: str | None = None
    source: str | Path = None
    pfm: str = None
    ver: str = None
    check_name: bool = True

    @pyd.root_validator(pre=True)
    def fill_missing(cls, values):
        if not (class_name := values.get('class_name', None)):
            return values  # leave for general validator to process the missing field

        kind = values.get('kind', None)
        parsed = AlgoEngine.parse_class_name(class_name, fail=False)

        if values.setdefault('check_name', True):
            if not parsed:
                ValueError(f"Mis-formed engine's {class_name=}")
            if kind and kind != parsed['kind']:
                ValueError(f"Engine's {kind=} mismatches that parsed form name {parsed['kind']}")

        if parsed:
            for key in ['name', 'pfm', 'ver', 'kind']:
                values.setdefault(key, parsed.get(key, None))
        return values

    def as_config_node(self):
        """return config as dict in the form suitable for saving in engines.yml"""
        cfg = self.dict(exclude_none=True, exclude_defaults=True, exclude_unset=True,
                        exclude={'name', 'pfm', 'source', 'ver', 'name'})
        return {cfg.pop('class_name'): cfg}


# enables importing engines from locations defined by the EnvLoc.ENGINES
PyModuleLocator.add_sys_path(*EnvLoc.ENGINES.existing(), check_exists=False)


@locatable(EnvLoc.ENGINES, pattern=ENGINES_YML)
class EnginesCatalogRM(ResourceModel):
    """
    Resource Model for engines registration files.
    """
    engines: dict[str, EngInfo]

    @pyd.validator('name', always=True)  # called first since 'name' comes from base class
    def default_name(cls, name, values):
        return name or str(values[cls._file_info_attr].folder)

    @pyd.validator('engines', pre=True)
    def normalize_engines(cls, engines: dict[str, dict], values: dict):
        """
        Validator for engines dict makes sure all the items are valid ``EngInfo`` objects.

         - fills `source` field from the model's yml file.
         - validates that module points to an existing file

        :param engines:
        :param values:
        :return:
        """
        file_info = values[cls._file_info_attr]

        try:
            loc = PyModuleLocator(file_info.folder)
        except ModuleNotFoundError as ex:
            _log.error(ex.msg)
            return {}

        def valid_info(name, info):
            _log.debug(f"Validating engine {name} {info}")
            if info.get('class_name', None) is None:
                info['class_name'] = name
            elif info.get('name', None) is None:
                info['name'] = name
            info['source'] = str(file_info.path)
            try:
                info['module'], info['package'] = loc.module_import_params(info['module'])
            except ModuleNotFoundError as ex:
                _log.error("Ignoring {class_name} engine in {source}: {ex}".format(**info, ex=ex))
                return None
            return EngInfo(**info)

        return {name: eng for name, info in engines.items()
                if (eng := valid_info(name, info))}

    def to_table(self):
        from toolbox.utils.pdtools import DataTable
        return DataTable(dict(v) for k, v in self.engines.items())


class Registry:
    # @formatter:off
    Categories = Literal['class_name', 'name', 'kind', 'pfm',
                         'module', 'package', 'source',
                         'ver', 'check_name']  # @formatter:on
    _Fields = Literal['kind', 'pfm']
    _categories = set(EngInfo.__fields__)
    assert _categories.issuperset(get_args(Literal))
    assert not (_ := _categories.symmetric_difference(get_args(Categories)))

    @staticmethod
    def discover_catalogs() -> list[EnginesCatalogRM]:
        """Discover and return list of found engines catalogs"""
        eng_rm = EnginesCatalogRM._manager
        return eng_rm.list('res') if eng_rm.discover() else []

    @staticmethod
    def _read_catalogs(*sources: EnginesCatalogRM):
        """Read catalogs from the given sources into a single DataFrame"""
        from pandas import concat, DataFrame
        return concat(cat.to_table() for cat in sources) if sources else DataFrame()

    def __repr__(self):
        if not (num := len(self._catalog)):
            return f"Engines catalog (empty!)"
        src_num = len(self._catalog.source.unique())
        return f"Engines catalog [{num}] from {src_num} sources."

    def __init__(self, *sources: Iterable[EnginesCatalogRM], discover=True):
        if discover:
            sources = [*self.discover_catalogs(), *sources]
        self._catalog = self._read_catalogs(*sources)

    def __len__(self):
        return self._catalog.__len__()

    def __getitem__(self, item):
        return self._catalog.iloc[item]

    @staticmethod
    def load_module_file(module_path, name=None):
        """Load module from its full path not assuming its accessibility by sys.path"""
        from importlib.util import spec_from_file_location, module_from_spec
        import sys

        if name is None:
            name = Path(module_path).stem

        spec = spec_from_file_location(name, module_path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def import_class(class_name, module, package=None, *, fail=True):
        """Import and return engine class with given `class_name` from given `module`.

         :param class_name: Name of the class defined in the `module`
         :param module: `module` - or full dotted package path
         :param package: optional package if `module` is relative (starts with dot)
         :param fail: if False return ``None`` installed of raising if failed
         :return: class of the engine or ``None``
         """

        def _error(err, msg=None, **kws):
            return error(err, msg=msg, logger=_log, fail=fail, **kws)

        try:
            m = import_module(module, package)  # ModuleNotFoundError
            return getattr(m, class_name)  # AttributeError
        except ModuleNotFoundError as ex:
            _error(ex, f"Invalid {module=} for engine '{class_name}'")
        except AttributeError:
            _error(NameError, f"Engine {class_name} not defined in module {module}")

    def list(self, fields: Categories | Collection[Categories] | Literal['all'] = (
            'class_name', 'kind', 'module', 'package')):
        """Return Table with specified fields of information about discovered engines.
        """
        if fields in ('all', None):
            fields = list(self._catalog.columns)
            fields.remove('ver')  # not used
        if not isinstance(fields, str):
            fields = list(fields)
        return self._catalog[fields] if self._catalog.size else []

    _OT = Literal['table', 'list', 'qualname', 'class', 'full']

    def find(self, name=None, *,
             out: _OT | Categories | Collection[Categories] = 'table',
             fail=True, **fields: dict[_Fields, str]):
        """
        Search for engines in the catalog based on the given criteria.

        This method searches for engines in a catalog according to specified fields and a name pattern.
        It returns the results in different formats based on the `out` parameter.

        Parameters:
        -----------
        name : str, optional
            The name pattern to match against engine names in the catalog. If provided,
            the search results are filtered based on this name.
        out : format fo the output, default 'table'

        fail : bool, optional
            If True, raises exceptions when errors occur; otherwise, suppresses them.
        **fields : dict, optional
            Additional info to filter the engines by.

        Returns:
        --------
        pandas.DataFrame, list, str, or class

        The search results formatted according to the `out` parameter:
         - 'table' - a ``DataFrame``-like object with columns `[class_name, kind, module, pfm]`
         - 'list' - a list of engine class names.
         - 'qualname' - the fully qualified name of the engine class
         - 'class' - the class itself
         - collection of attributes to return in the table
         - a field to return from the table as ``Series``

        Raises:
        -------
        KeyError
            If provided invalid info fields.
        NameError
            If no engines match the search criteria.
        ModuleNotFoundError
            If the specified module cannot be found when out='class'.
        AttributeError
            If the specified engine class is not found in its smodule.
        ValueError
            If the `out` parameter is set to an invalid value.

        Examples:
        ---------
        >>> engines.find(name='my_engine', out='list', pfm='T')
        ['MyEngine1', 'MyEngine2']

        >>> engines.find('MyEngine', out='qualname')
        'package.module.MyEngine'
        """

        def _error(err, msg=None, **kws):
            return error(err, msg=msg, logger=_log, fail=fail, **kws)

        # -------- validate inputs -----------
        if self._catalog.empty:
            return error(RuntimeError, "Engines catalog is empty!")

        if not (fields or name):
            return _error(ValueError, "Engines find query has no arguments!")

        if invalid_fields := (set(fields) - self._categories):
            raise KeyError(f"{invalid_fields=}, valid catalog fields: {self._categories}")

        # - filter catalog  - first by fields
        cat = self._catalog
        if fields := drop_undef(ns=fields):
            if None in set(fields.values()):
                from pandas import NA
                fields = {k: NA if v is None else v for k, v in fields.items()}

            cat = cat.set_index(list(cat.columns.difference(['class_name'])))
            cat = cat.qix(**fields).reset_index()[self._catalog.columns]  # qix operates only on the index!
            if cat.empty:
                return _error(LookupError, f"Not found engines with {fields=}")
        # - filter now by the name (here cat is NOT empty)
        if name:
            sel = cat.class_name.apply(lambda _: name.lower() in _.lower())
            cat = cat[sel]

        if cat.empty:
            return _error(NameError, f"Not found engines find({name=} {fields=})")

        match out:
            case 'qualname' | 'class':
                if (n := len(cat)) != 1:
                    return _error(NameError, f"'{name} matches {n} engines instead expected 1!")
                res = cat.iloc[0]
                if out == 'qualname':
                    return f"{res.package or ''}{res.module}.{res.class_name}"
                return self.import_class(res.class_name, res.module, res.package, fail=fail)
            case 'table':
                return cat[['class_name', 'kind', 'module', 'package', 'pfm']]
            case 'full':
                return cat
            case list(columns):
                return cat[columns]
            case 'list':
                return list(cat.class_name)
            case _:
                if isinstance(out, str) and out in self._categories:
                    return cat[out]
                raise ValueError(f"Invalid argument 'out'={out}")

    def klass(self, name, **info) -> type[AlgoEngine]:
        """Return engine class given a portion of its name and
        optionally additional query on any fields from {'kind', 'pfm', 'ver'}.

        Raise ``LookupError`` if failed to find and import exactly **one** such class.

        :param name: Any portion of the class name
        :param info: narrow search by specifying additional engine information
        """
        try:
            return self.find(name, out='class', **info)
        except BaseException as ex:
            raise LookupError(str(ex))

    @staticmethod
    def create_catalog(path: Path | str = None, *, package: str = None,
                       relative=True, deep=0, overwrite=False):
        """
        Create engines catalog file for engines defined in the given `source`, which may point to
         - a python module file (.py), then look for engine classes there, or
         - a folder, then collect from all the modules found there.

        The catalog is created in the specified path.

        Only one `path` or `package` name (dotted) must be provided

        :param path:   folder to search
        :param package: package to search
        :param relative: write relative modules
        :param deep: number of levels under the path to search (use big number for 'all')
        :param overwrite: otherwise fails if file exist
        :return:
        """
        import shutil
        from toolbox.param import TBox
        # ---------- if its a package provided, convert into path first ---------------
        if package:  # may be it's a dotted package name
            assert not path, "Use either folder or module!"
            path = Path(import_module(package).__file__)
            if path.name == '__init__.py':  # package name, not a specific module
                path = path.parent
        else:
            path = Path(path).resolve(strict=True)

        # -------------  find all the python modules files in the folder --------------

        if path.is_file():
            files = [path]
            folder = path.parent
        else:
            folder = path
            files = list(PyModuleLocator.modules_under(path, sys_path=True, deep=deep))
            files or _log.error(f'No modules found in {str(folder)}')

        # ------------ find all the Engine classes in those modules -------------------
        def engine_configs() -> Iterable[EngInfo]:
            loc = PyModuleLocator(folder)

            for file in files:
                full_module = loc.file_to_module(file, relative=False)
                module = loc.file_to_module(file, relative=True) if relative else full_module
                _log.debug(f"Searching for engines in {full_module}")
                try:
                    module_obj = import_module(full_module)
                    namespace = vars(module_obj)
                except BaseException as ex:
                    namespace = {}
                    _log.error(f"Error importing {full_module}: {ex}")
                eng_num = 0
                for name, obj in namespace.items():
                    if isinstance(obj, type) and issubclass(obj, AlgoEngine) and not obj.is_abstract:
                        _log.debug(f"Engine class found: {obj.__name__}")
                        eng_num += 1
                        yield EngInfo(class_name=obj.__name__,
                                      kind=obj.kind,
                                      check_name=obj._check_name,
                                      module=module)
                eng_num and _log.info(f"Found {eng_num} engines in {full_module}")

        # ------------------- Create engines catalog file in the folder ---------------

        if (yml_file := folder / ENGINES_YML).is_file() and not overwrite:
            raise FileExistsError(f"Engines Catalog already exists: {str(yml_file)}")

        cfg = TBox(engines={})
        for eng in engine_configs():
            cfg.engines.update(eng.as_config_node())  # FixMe Fail on same engine name
        num = len(cfg['engines'])

        # ---- before overwriting previous validate the new one ------
        tmp_file = yml_file.with_stem('_engines')  # to postpone
        _log.debug(f"Validating temporal catalog with {num} engines {str(tmp_file)}")
        cfg.to_yaml(tmp_file)

        cat = EnginesCatalogRM.from_config(tmp_file)  # simulate the usage
        if len(Registry(cat, discover=False)) != num:
            raise AssertionError(f"Failed to validate created catalog {str(tmp_file)}")

        shutil.move(tmp_file, yml_file)
        _log.info(f"Altogether {num} engines recorded into {str(yml_file)}.")
        return num


engines = Registry()


def engine_class(name, **info):
    """Return engine class given a portion of its name and
    optionally additional query on any fields from {'kind', 'pfm', 'ver'}.

    Raise ``LookupError`` if failed to find and import exactly **one** such class.

    :param name: Any portion of the class name
    :param info: narrow search by specifying additional engine information
    """
    return engines.klass(name, **info)

