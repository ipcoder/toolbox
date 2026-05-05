import logging
import os
from pathlib import Path
import dotenv

from .filesproc import Locator

env_log = logging.getLogger('env')

# make sure WARNING + levels of env log ALSO appear in stderr
if not env_log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel('WARNING')
    env_log.addHandler(handler)

#  -------------------- Environment variables names --------------------------
# Those variables can be set in process environment to
# change EnvLoc initialization behaviour

ENV_FILE_VAR = 'ALG_DOTENV_FILE'  # change path to DEFAULT dotenv file (algodev/.env)
ENV_OVER_VAR = 'ALG_DOTENV_OVER'  # IF set to 'true' dotenv file OVERRIDE system variables
TEMP_VAR = 'TEMP'  # if not defined tempfile.gettempdir() is used
# init temporal folder
if not os.getenv(TEMP_VAR):
    import tempfile

    os.environ[TEMP_VAR] = tempfile.gettempdir()

# default behaviour if ALG_DOTENV_OVER is not set
def _env_override_policy(override, default='false'):
    """
    Applies decisions policy for determining dotenv environment override behaviour.

    Note, that override=True means that .env overrides the system environment

    - if `True` | `False` - defines the override explicitely
    - if `None`:
        - read override from `ALG_DOTENV_OVER` environment variable if defined
        - use `default` if not defined

    :param override: the value to start from
    :param default: default value for `ALG_DOTENV_OVER` if not set by environment
    :return:
    """
    # set `ovveride` value from the argument or envar ENV_OVER_VAR
    if override not in (True, False, None):
        raise ValueError(f"Invalid value for {override=}")

    if override is None:
        override = os.getenv(ENV_OVER_VAR, default).lower()
        if override in {'yes', 'true', '1'}:
            override = True
        elif override in {'no', 'false', '0'}:
            override = False
        else:
            raise ValueError(f"Invalid value of {ENV_OVER_VAR}={override}")
        env_log.debug(f'dotenv {override=} sys env ({ENV_OVER_VAR}={os.getenv(ENV_OVER_VAR)})')
    return override


class EnvLoc:
    """Singleton-style bundle of ``Locator``s for algutils-wide paths.

    Customizable per working environment using ``reset``.
    """
    # LOCAL_DATA = Locator(envar='ALG_LOCAL_DATA')
    # NET_DATA = Locator(envar='ALG_NET_DATA', alarm=FileExistsError)
    # DATA = Locator(envar='ALG_DATA', alarm=FileExistsError)
    # DATASETS = Locator(envar='ALG_DATASETS', alarm=_log.error)
    # RESOURCES = Locator(envar='ALG_RESOURCES')

    _last_set = None  # dotenv file used to set the current state
    _initial_envars = None  # initial envars from os.environment used by locators

    @classmethod
    def safe_locators(cls, status=None):
        """
        If status is `None` print *safe* attribute of the locators.

        Otherwise set all the locators to the given `status` which must be bool or 0 < timeout < 10.

        Individual locators safe attributes can be set directly.
        """
        if status is None:
            for name, loc in cls.locators().items():
                print(f"{name:>20s}\t{loc.safe:>6}")
        elif isinstance(status, bool) or \
                isinstance(status, (int, float)) and 0 < status < 5:
            for loc in cls.locators().values():
                loc.safe = status
        else:
            raise ValueError(f"Invalid {status=}")

    @classmethod
    def last_set(cls) -> str | None:
        """Return last set environment, ``None`` if never set.

        Environment is usually set using ``env.setup()`` or ``env.EnvLoc.reset()``
        """
        return cls._last_set

    @classmethod
    def locators(cls) -> dict[str, Locator]:
        return {name: loc for name, loc in cls.__dict__.items() if isinstance(loc, Locator)}

    @classmethod
    def repr(cls, existing=False) -> str:
        """
        String representation of the current EnvLoc state
        :return:
        """
        def max_line_len(strings):
            """Caclulate length of mazimal line (\n separated) among all, the strings"""
            lines = (line for s in strings for line in s.split('\n'))
            return len(max(lines, key=len))

        locs = cls.locators()
        fmt = f'{{:>{max_line_len(locs) + 2}s}}: {{}}'

        locs = [fmt.format(name, loc.repr(existing)) for name, loc in locs.items()]
        head = f" Current EnvLoc (dotfile: {cls._last_set})"
        locs_len, head_len = max_line_len(locs), len(head)
        if head_len < locs_len:
            head += ' ' * (locs_len - head_len)
            head_len = locs_len
        sep = '─' * head_len
        return '\n'.join(['┌' + sep + '┐', '│' + head + '│', '└' + sep + '┘', *locs])

    @classmethod
    def validate(cls, alarm: Locator.AlarmTypes | None = None):
        """
        Run validation on all the locators with given alarm
        (``None`` means using their own alarms set in constructor)
        :param alarm:
        :return:
        """
        for loc in cls.locators().values():
            # loc.set_check_opt()
            loc.validate(alarm)

    @classmethod
    def reset(cls, env_path=None, override=None, validate: bool=True) -> bool:
        """Reset locators after loading from env file described by either:
          - explicit path to the env file
          - folder to start loking from (by climbing up)
          - `None` the try in this order:
            - ``ALG_DOTENV_FILE`` environment variable
            - find starting up from the ``CWD``
            - finally default `.env` under the ``algutils`` package (``src/algutils``)

        :param env_path: full or relative path, name in the tree above, `None` for default ``.env``
        :param override: ``True`` override envars by .env content.
                         ``None`` - follow  ``ALG_DOTENV_OVER`` envar.
        :param validate: if `True` check and log if locators point to at least one actual folder
        :return: `True` if environment was reset, `False` if not found
        """
        # optional absolute or relative path to env file - if not given, uses default .env in the project

        cls.reset_os_environ()
        if not (env_path := cls._locate_dotenv(env_path, fail=validate)):
            return False
        
        cls._process_dotenv(env_path, override)
        cls._last_set = str(Path(env_path).absolute())  # keep last set environment

        for loc in cls.locators().values():
            loc.safe = bool(loc.alarm)

        if validate:
            cls.validate()
        return True

    @classmethod
    def _process_dotenv(cls, env_file, override=None):
        from .filesproc import represents_path, normalize

        def same_value(v1, v2):
            """Return true if two values are the same.
            If they represent paths normalize before comparing.
            """
            if v1 and v2 and any(map(represents_path, [v1, v2])):
                return normalize(v1) == normalize(v2)
            return v1 == v2

        override = _env_override_policy(override)
        env_log.warning(f"Set environment from '{env_file}', {override=} vars in sys env")

        # validate that `env_file` contains only expected vars (defined in `cls`)
        expected = {loc.envar for loc in cls.locators().values()}
        denvars = {n: v for n, v in dotenv.dotenv_values(env_file).items()}
        from_denv = {n for n, v in denvars.items() if v is not None}
        if unknown := from_denv - expected:
            msg = f"Some env vars in {env_file} are {unknown = }"
            env_log.error(msg)
            raise NameError(msg)

        # find vars defined in both os.environment AND in `env_file`
        # and INFORM about which will be selected according to `override`
        if redefined := expected.intersection(os.environ, from_denv):
            side, act = ('←', 'overridden by') if override else ('→', 'override')
            if redefined := [
                f"{n}: {enval} {side} {denval}" for n in redefined
                if not same_value(enval := os.getenv(n), denval := denvars[n])
            ]:
                env_log.warning(f"Envars {act} dotenv file {env_file}:\n\t{', '.join(redefined)}")
        dotenv.load_dotenv(env_file, verbose=True, override=override)

    @staticmethod
    def _locate_dotenv(env_path, fail=True):
        """Locate dot env file given folderm file name or full path.

        :param env_path: file name | full path | folder containing .env
        :return: full path to located file or raise ``FileNotFoundError``
        """

        def find_dotenv_dir(folder, filename='.env', raise_error_if_not_found=False):
            """Runs find_doten from given folder"""
            cwd = os.getcwd()
            os.chdir(folder)
            found_env = dotenv.find_dotenv(
                filename,
                raise_error_if_not_found=raise_error_if_not_found
            )
            os.chdir(cwd)
            return found_env

        if env_path:  # hint provided: folder | file_name | full_path
            if (_path := Path(env_path)).is_dir():
                env_log.debug('Looking .env in from requested folder {_path}...')
                env_path = find_dotenv_dir(env_path, raise_error_if_not_found=True)
            elif not _path.is_file():
                env_path = dotenv.find_dotenv(env_path, raise_error_if_not_found=True)
        elif env_path := os.getenv(ENV_FILE_VAR, None):  # not provided - try to read from environmet
            if not Path(env_path).is_file():
                if not fail: return None        
                raise FileNotFoundError(f'Not found dotenv defined in {ENV_FILE_VAR}="{env_path}"')
            env_log.debug(f'dotenv set by the environment variable {ENV_FILE_VAR}={env_path}')
        elif not (env_path := dotenv.find_dotenv(usecwd=True)):  # try first search from the current dir
            msg = f'Getting default .env after not found at CDW: {os.getcwd()}'
            env_log.debug(msg)
            if not fail: return None
            raise FileNotFoundError(msg)
        return env_path

    @classmethod
    def reset_os_environ(cls, **fixes):
        """
        Ensures that os.environ variables used by locators are reset to their intital state.

        Requires since locators keep environment variables names, but not values,
        they are reading `os.environ` every time location is requested.
        That makes them sensitive to changes in those values.

        As a hacking tool, the initial system environment may be changed with `fixes`.
        """
        if cls._initial_envars is None:  # save initial values
            cls._initial_envars = {
                loc.envar: os.getenv(loc.envar)
                for loc in cls.locators().values() if loc.envar
            }
            cls._initial_envars.update(fixes)  # not really can happen here
        else:  # restore
            cls._initial_envars.update(fixes)  # fix the "initial" state
            for k, v in cls._initial_envars.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# EnvLoc.reset(validate=False)  # set by environnet variable or default
