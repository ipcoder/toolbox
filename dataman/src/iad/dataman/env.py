from iad.core.env import EnvLoc as CoreEnvLoc, Locator, env_log


class EnvLoc(CoreEnvLoc):
    """Environment locators for dataman package."""
    DATA = Locator(envar='ALG_DATA', alarm=env_log.error)
    DATASETS = Locator(envar='ALG_DATASETS', alarm=env_log.error)
    RESOURCES = Locator(envar='ALG_RESOURCES', alarm=env_log.error)
 
EnvLoc.reset(validate=False)
