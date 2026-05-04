from iad.core.env import EnvLoc as CoreEnvLoc, Locator

class EnvLoc(CoreEnvLoc):
    """Environment locators for annotations package."""
    ISSUES = Locator('/tmp/issues/', envar='ALG_ISSUES')  # ToDo: issues loaction
    

    

